"""
Retriever module - 하이브리드 검색 시스템

메타데이터 필터링과 시맨틱 검색을 결합한 하이브리드 검색을 제공합니다.
논문 Section 4.4 (RAG) 및 4.4.2 (Hybrid Retrieval Strategy)를 구현합니다.

핵심 전략:
1. 시맨틱 검색 우선 (임베딩 기반 유사도)
2. 신뢰도 임계값 미달시 BM25 폴백
3. 메타데이터 필터링으로 검색 범위 축소
"""

import json
from typing import List, Dict, Optional, Tuple
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD
)
from .ingest import get_embedding_model, initialize_vector_store
from .utils import (
    build_metadata_filter,
    clean_query_for_embedding,
    extract_teams_from_query
)


# =============================================================================
# 벡터 스토어 접근
# =============================================================================

_vector_store: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    """
    벡터 스토어 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        Chroma: 벡터 스토어 인스턴스
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = initialize_vector_store(
            documents=None,  # 기존 스토어 로드
            persist_directory=str(CHROMA_DB_DIR),
            collection_name=COLLECTION_NAME
        )
    return _vector_store


# =============================================================================
# 시맨틱 검색
# =============================================================================

def semantic_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    metadata_filter: Dict = None
) -> List[Tuple[Document, float]]:
    """
    시맨틱 검색을 수행합니다.
    
    임베딩 유사도 기반으로 관련 문서를 검색합니다.
    Instruct 모델 형식으로 쿼리를 변환하여 검색 정확도를 높입니다.
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 최대 문서 수
        metadata_filter: ChromaDB where 필터
    
    Returns:
        List[Tuple[Document, float]]: (문서, 유사도 점수) 목록
    """
    vector_store = get_vector_store()
    
    # Instruct 모델용 쿼리 포맷팅
    # 논문 Section 4.4.1에서 검증된 형식
    formatted_query = (
        f"Instruct: Find the baseball dataset covering the given team(s) and statistics. "
        f"Query: {query}"
    )
    
    # 메타데이터 필터가 있는 경우
    if metadata_filter:
        results = vector_store.similarity_search_with_relevance_scores(
            query=formatted_query,
            k=top_k,
            filter=metadata_filter
        )
    else:
        results = vector_store.similarity_search_with_relevance_scores(
            query=formatted_query,
            k=top_k
        )
    
    return results


def get_all_documents_for_bm25() -> List[Document]:
    """
    BM25 검색을 위해 모든 문서를 로드합니다.
    
    Returns:
        List[Document]: 모든 문서 목록
    """
    vector_store = get_vector_store()
    
    # ChromaDB에서 모든 문서 조회
    collection = vector_store._collection
    results = collection.get(include=["documents", "metadatas"])
    
    documents = []
    for doc_text, metadata in zip(results["documents"], results["metadatas"]):
        documents.append(Document(
            page_content=doc_text,
            metadata=metadata or {}
        ))
    
    return documents


# =============================================================================
# 하이브리드 검색
# =============================================================================

def hybrid_search(
    query: str,
    teams: List[str] = None,
    data_type: str = None,
    date: str = None,
    top_k: int = RETRIEVAL_TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[Optional[Document], float, str]:
    """
    하이브리드 검색을 수행합니다.
    
    논문 Section 4.4.2 전략:
    1. 시맨틱 검색으로 상위 K개 후보 검색
    2. 최고 점수가 임계값 이상이면 해당 결과 반환
    3. 임계값 미달시 BM25 앙상블 폴백
    
    Args:
        query: 검색 쿼리
        teams: 필터링할 팀 목록 (정규화된 영문명)
        data_type: 데이터 유형 ("season" 또는 "match")
        date: 경기 날짜 (match_analysis인 경우)
        top_k: 검색할 최대 문서 수
        similarity_threshold: 시맨틱 검색 신뢰도 임계값
    
    Returns:
        Tuple[Optional[Document], float, str]: 
            (검색된 문서, 신뢰도 점수, 검색 방법)
    
    Example:
        >>> doc, score, method = hybrid_search(
        ...     query="한화 시즌 성적",
        ...     teams=["Hanwha"],
        ...     data_type="season"
        ... )
        >>> print(f"검색 방법: {method}, 점수: {score:.2f}")
    """
    # 1. 메타데이터 필터 구성
    metadata_filter = build_metadata_filter(
        teams=teams,
        data_type=data_type,
        date=date
    )
    
    # 2. 검색 쿼리 정제
    cleaned_query = clean_query_for_embedding(query, teams, date)
    
    print(f"🔍 검색 쿼리: {cleaned_query}")
    if metadata_filter:
        print(f"📋 메타데이터 필터: {metadata_filter}")
    
    # 3. 시맨틱 검색 수행
    semantic_results = semantic_search(
        query=cleaned_query,
        top_k=top_k,
        metadata_filter=metadata_filter
    )
    
    # 결과가 없는 경우
    if not semantic_results:
        print("⚠️ 시맨틱 검색 결과 없음, 필터 없이 재검색...")
        # 필터 없이 재시도
        semantic_results = semantic_search(
            query=cleaned_query,
            top_k=top_k,
            metadata_filter=None
        )
        
        if not semantic_results:
            return None, 0.0, "no_results"
    
    # 4. 최고 점수 확인
    top_doc, top_score = semantic_results[0]
    
    print(f"📊 시맨틱 검색 최고 점수: {top_score:.4f}")
    
    # 5. 임계값 체크
    if top_score >= similarity_threshold:
        # 시맨틱 검색 결과 사용
        return top_doc, top_score, "semantic"
    
    # 6. 폴백: 앙상블 검색 (시맨틱 + BM25)
    print(f"⚠️ 점수 {top_score:.4f} < 임계값 {similarity_threshold}, 앙상블 폴백 수행")
    
    try:
        # BM25 검색기 구성
        all_docs = get_all_documents_for_bm25()
        
        if len(all_docs) < 2:
            # 문서가 적으면 시맨틱 결과 그대로 사용
            return top_doc, top_score, "semantic_fallback"
        
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = top_k
        
        # 시맨틱 검색기 래핑
        vector_store = get_vector_store()
        semantic_retriever = vector_store.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": metadata_filter if metadata_filter else None
            }
        )
        
        # 앙상블 검색기 (논문: 시맨틱 0.8, BM25 0.2)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.8, 0.2]
        )
        
        ensemble_results = ensemble_retriever.invoke(cleaned_query)
        
        if ensemble_results:
            return ensemble_results[0], top_score, "ensemble"
        else:
            return top_doc, top_score, "semantic_fallback"
            
    except Exception as e:
        print(f"⚠️ 앙상블 검색 실패: {e}, 시맨틱 결과 사용")
        return top_doc, top_score, "semantic_fallback"


# =============================================================================
# 검색 결과 후처리
# =============================================================================

def extract_raw_data(document: Document) -> Dict:
    """
    Document에서 원본 JSON 데이터를 추출합니다.
    
    Args:
        document: LangChain Document
    
    Returns:
        Dict: 원본 JSON 데이터
    """
    raw_data_str = document.metadata.get("raw_data", "{}")
    
    try:
        return json.loads(raw_data_str)
    except json.JSONDecodeError:
        return {}


def prepare_context_for_llm(
    document: Document,
    query_teams: List[str] = None
) -> Dict:
    """
    LLM 컨텍스트용 데이터를 준비합니다.
    
    논문 Section 4.4.3 (Post-Retrieval Data Preparation):
    - 불필요한 데이터 제거
    - 요청된 팀 데이터만 필터링
    - 헤더 정보 제거로 토큰 절약
    
    Args:
        document: 검색된 Document
        query_teams: 쿼리에서 추출된 팀 목록 (필터링용)
    
    Returns:
        Dict: 정제된 컨텍스트 데이터
    """
    raw_data = extract_raw_data(document)
    
    # 불필요한 필드 제거
    fields_to_remove = ["headers", "_source_file", "_loaded_at"]
    for field in fields_to_remove:
        raw_data.pop(field, None)
    
    # 팀 필터링 (시즌 데이터에서 특정 팀만 추출)
    if query_teams and document.metadata.get("type") == "season":
        # 선수 데이터에서 요청된 팀 선수만 필터링
        if "players" in raw_data and isinstance(raw_data["players"], list):
            from rapidfuzz import fuzz
            
            filtered_players = []
            for player in raw_data["players"]:
                player_team = player.get("team", "")
                
                # 팀명 매칭 검사
                for query_team in query_teams:
                    if fuzz.QRatio(player_team.lower(), query_team.lower()) >= 60:
                        filtered_players.append(player)
                        break
            
            raw_data["players"] = filtered_players
            raw_data["_filtered_for_teams"] = query_teams
    
    return {
        "type": document.metadata.get("type"),
        "teams": document.metadata.get("teams", []),
        "date": document.metadata.get("date"),
        "season": document.metadata.get("season"),
        "data": raw_data
    }


# =============================================================================
# 통합 검색 인터페이스
# =============================================================================

def retrieve_for_query(
    query: str,
    query_type: str,
    teams: List[str] = None,
    date: str = None
) -> Tuple[Optional[Dict], float, str]:
    """
    쿼리 유형에 맞는 검색을 수행하고 결과를 반환합니다.
    
    chain.py에서 호출하는 통합 검색 인터페이스입니다.
    
    Args:
        query: 사용자 쿼리
        query_type: 쿼리 유형 ("season_analysis" 또는 "match_analysis")
        teams: 정규화된 팀 목록
        date: 경기 날짜 (match_analysis인 경우)
    
    Returns:
        Tuple[Optional[Dict], float, str]:
            (컨텍스트 데이터, 신뢰도 점수, 검색 방법)
    """
    # 데이터 타입 결정
    if query_type == "match_analysis":
        data_type = "match"
    elif query_type == "season_analysis":
        data_type = "season"
    else:
        data_type = None  # 필터 없음
    
    # 하이브리드 검색 수행
    document, score, method = hybrid_search(
        query=query,
        teams=teams,
        data_type=data_type,
        date=date
    )
    
    if document is None:
        return None, 0.0, method
    
    # 컨텍스트 데이터 준비
    context = prepare_context_for_llm(document, teams)
    
    return context, score, method


# =============================================================================
# 편의 함수 (기존 스켈레톤 호환)
# =============================================================================

def keyword_search(query: str, top_k: int = 5):
    """
    키워드 기반 BM25 검색 (레거시 호환용)
    """
    all_docs = get_all_documents_for_bm25()
    
    if not all_docs:
        return []
    
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = top_k
    
    return bm25_retriever.invoke(query)


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    # 테스트 쿼리
    test_cases = [
        {
            "query": "한화 시즌 성적",
            "query_type": "season_analysis",
            "teams": ["Hanwha"]
        },
        {
            "query": "LG 두산 경기",
            "query_type": "match_analysis",
            "teams": ["LG", "Doosan"]
        },
    ]
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"쿼리: {case['query']}")
        print(f"{'='*60}")
        
        context, score, method = retrieve_for_query(
            query=case["query"],
            query_type=case["query_type"],
            teams=case.get("teams")
        )
        
        if context:
            print(f"✅ 검색 성공 ({method})")
            print(f"   점수: {score:.4f}")
            print(f"   타입: {context['type']}")
            print(f"   팀: {context['teams']}")
        else:
            print(f"❌ 검색 실패: {method}")
