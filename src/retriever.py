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

from openai import OpenAI

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
# 쿼리 번역 (한국어 → 영어)
# =============================================================================

_openai_client: OpenAI = None


def get_openai_client() -> OpenAI:
    """OpenAI 클라이언트 싱글톤"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def translate_query_to_english(query: str) -> str:
    """
    한국어 쿼리를 완전히 영어로 번역합니다.
    
    선수 이름, 팀 이름, 야구 용어 모두 영어로 변환합니다.
    Description도 영어로 저장되어 있어 영어-영어 비교로 유사도가 높아집니다.
    
    Args:
        query: 한국어 쿼리
    
    Returns:
        str: 완전히 영어로 번역된 쿼리
    """
    client = get_openai_client()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator for KBO (Korean Baseball Organization) queries. "
                    "Translate the Korean query completely to English:\n"
                    "1. Romanize player names (e.g., 류현진 → Ryu Hyun-jin, 후라도 → Hueraldo)\n"
                    "2. Translate team names (e.g., 한화 → Hanwha, 롯데 → Lotte)\n"
                    "3. Translate baseball terms (e.g., 시즌 성적 → season stats, 방어율 → ERA)\n"
                    "4. Remove adjectives/filler words (e.g., '진짜 잘하는' → remove)\n"
                    "5. Output ONLY the translated query, nothing else.\n\n"
                    "Examples:\n"
                    "- '류현진 2025 시즌 성적' → 'Ryu Hyun-jin 2025 season stats'\n"
                    "- '진짜 잘하는 후라도 2025시즌 성적' → 'Hueraldo 2025 season stats'\n"
                    "- '한화 이글스 올시즌 투수 분석' → 'Hanwha Eagles this season pitching analysis'"
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()


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
    metadata_filter: Dict = None,
    verbose: bool = False
) -> List[Tuple[Document, float]]:
    """
    시맨틱 검색을 수행합니다.
    
    임베딩 유사도 기반으로 관련 문서를 검색합니다.
    한국어 쿼리를 영어로 번역하여 영어 description과 비교합니다.
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 최대 문서 수
        metadata_filter: ChromaDB where 필터
        verbose: 번역 결과 출력 여부
    
    Returns:
        List[Tuple[Document, float]]: (문서, 유사도 점수) 목록
    """
    vector_store = get_vector_store()
    
    # 한국어 쿼리를 영어로 번역 (영어 description과 비교하기 위해)
    translated_query = translate_query_to_english(query)
    
    if verbose:
        print(f"📝 원본 쿼리: {query}")
        print(f"🔄 번역 쿼리: {translated_query}")
    
    # Instruct 모델용 쿼리 포맷팅
    # 논문 Section 4.4.1에서 검증된 형식
    formatted_query = (
        f"Instruct: Find the baseball dataset covering the given team(s) and statistics. "
        f"Query: {translated_query}"
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
    
    # 2. 검색 쿼리 정제 (논문 Section 4.3.2: Fully Cleaned 쿼리)
    # data_type을 query_type으로 변환
    query_type_map = {"season": "season_analysis", "game": "match_analysis"}
    query_type = query_type_map.get(data_type)
    
    cleaned_query = clean_query_for_embedding(query, teams, date, query_type)
    
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
    
    # 3.5. 팀 필터링 후처리 (ChromaDB 필터와 중복 확인용)
    if teams:
        filtered_results = []
        for doc, score in semantic_results:
            home_team = doc.metadata.get("home_team", "")
            away_team = doc.metadata.get("away_team", "")
            doc_teams = {home_team, away_team}
            
            # 요청된 모든 팀이 home_team 또는 away_team에 있는지 확인
            if all(team in doc_teams for team in teams):
                filtered_results.append((doc, score))
        
        if filtered_results:
            semantic_results = filtered_results
            print(f"📋 팀 필터 후처리 적용: {len(filtered_results)}개 결과")
        else:
            print(f"⚠️ 팀 필터 후 매칭 결과 없음, 원본 결과 유지")
    
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
# 검색 결과 후처리 (Post-Retrieval Data Preparation)
# 논문 Section 4.4.3: 토큰 절약을 위한 데이터 정제
# =============================================================================

def extract_raw_data(document: Document) -> Dict:
    """
    Document에서 원본 JSON 데이터를 추출합니다.
    
    Args:
        document: LangChain Document
    
    Returns:
        Dict: 원본 JSON 데이터
    """
    # 메타데이터에서 원본 데이터 추출 (original_data 또는 raw_data)
    raw_data_str = document.metadata.get("original_data") or document.metadata.get("raw_data", "{}")
    
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
    - clean_opponent: 요청된 팀 데이터만 필터링 (QRatio >= 60)
    - clean_game: 헤더 정보 제거
    - 'headers' 속성 제거로 약 600 토큰 절약
    
    Args:
        document: 검색된 Document
        query_teams: 쿼리에서 추출된 팀 목록 (필터링용)
    
    Returns:
        Dict: 정제된 컨텍스트 데이터
    """
    raw_data = extract_raw_data(document)
    data_type = document.metadata.get("type")
    
    # =================================================================
    # 논문 4.4.3: 'headers' 속성 제거 (약 600 토큰 절약)
    # =================================================================
    raw_data.pop("headers", None)
    
    # =================================================================
    # 논문 4.4.3: clean_opponent - 팀 필터링 (시즌 데이터)
    # QRatio < 60인 선수 데이터 제거, >= 60인 경우 LLM이 최종 판단
    # =================================================================
    if data_type == "season" and query_teams:
        from rapidfuzz import fuzz
        
        if "players" in raw_data and isinstance(raw_data["players"], list):
            filtered_players = []
            for player in raw_data["players"]:
                player_team = player.get("team", "")
                for query_team in query_teams:
                    # QRatio >= 60: 유지 (LLM이 최종 판단)
                    # QRatio < 60: 제거 (확실히 다른 팀)
                    if fuzz.QRatio(player_team.lower(), query_team.lower()) >= 60:
                        filtered_players.append(player)
                        break
            raw_data["players"] = filtered_players
        
        raw_data["_filtered_for_teams"] = query_teams
    
    # =================================================================
    # 논문 4.4.3: clean_game - 중복 헤더 정보 제거 (경기 데이터)
    # =================================================================
    # 헤더는 이미 위에서 제거됨
    
    # teams 메타데이터 파싱 (game 데이터는 home_team, away_team으로 저장됨)
    home_team = document.metadata.get("home_team", "")
    away_team = document.metadata.get("away_team", "")
    
    if home_team and away_team:
        # game 데이터: home_team, away_team에서 팀 목록 구성
        teams_list = [home_team, away_team]
    else:
        # season 데이터: team 필드 사용
        team = document.metadata.get("team", "")
        teams_list = [team] if team else []
    
    return {
        "type": document.metadata.get("type"),
        "teams": teams_list,
        "home_team": home_team,
        "away_team": away_team,
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
    # 데이터 타입 결정 (DB에 저장된 타입명과 일치시킴)
    if query_type == "match_analysis":
        data_type = "game"  # DB에는 "game"으로 저장됨
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
