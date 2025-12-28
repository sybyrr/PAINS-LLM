"""
Ingest module - 데이터 적재 파이프라인

JSON 데이터를 로드하고 ChromaDB에 적재하는 기능을 제공합니다.
논문 Section 4.2 (Embedding the data) 전략을 구현합니다.

핵심 전략:
1. JSON 내용을 그대로 임베딩하지 않음
2. 각 파일에 대해 "설명 문장(Descriptive Sentence)"을 생성하여 page_content로 저장
3. 원본 JSON 데이터는 metadata 필드에 저장
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from tqdm import tqdm

from .config import (
    CHROMA_DB_DIR, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL,
    SEASON_DATA_DIR,
    MATCH_DATA_DIR
)
from .utils import generate_descriptive_sentence, TEAM_EN_TO_KO


# =============================================================================
# 임베딩 모델 초기화
# =============================================================================

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    임베딩 모델을 초기화합니다.
    
    논문에서 검증된 multilingual-e5-large-instruct 모델을 사용합니다.
    L2 정규화를 적용하여 코사인 유사도 계산에 최적화합니다.
    
    Returns:
        HuggingFaceEmbeddings: 초기화된 임베딩 모델
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # GPU 없는 환경 지원
        encode_kwargs={
            "normalize_embeddings": True  # L2 정규화 - 논문 권장
        }
    )


# =============================================================================
# 데이터 로딩 함수
# =============================================================================

def load_season_data(data_dir: Path = None) -> List[Dict]:
    """
    시즌 누적 데이터(Global Dataset)를 로드합니다.
    
    파일명 예시: KBO_2025_Season_Total.json, KBO_2025_Hanwha.json
    
    Args:
        data_dir: 시즌 데이터 디렉토리 경로
    
    Returns:
        List[Dict]: 로드된 시즌 데이터 목록
    """
    if data_dir is None:
        data_dir = SEASON_DATA_DIR
    
    season_data = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️ 시즌 데이터 디렉토리가 존재하지 않습니다: {data_path}")
        return season_data
    
    json_files = list(data_path.glob("*.json"))
    print(f"📂 시즌 데이터 파일 발견: {len(json_files)}개")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 메타데이터 추가
            data["_source_file"] = json_file.name
            data["_data_type"] = "season"
            data["_loaded_at"] = datetime.now().isoformat()
            
            # 파일명에서 팀/시즌 정보 추출 시도
            filename = json_file.stem  # 확장자 제외
            
            # 시즌 정보 추출 (예: KBO_2025_Hanwha)
            season_match = re.search(r'(\d{4})', filename)
            if season_match:
                data["season"] = season_match.group(1)
            
            season_data.append(data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
        except Exception as e:
            print(f"❌ 파일 로드 오류 ({json_file.name}): {e}")
    
    return season_data


def load_match_data(data_dir: Path = None) -> List[Dict]:
    """
    개별 경기 데이터(Match Dataset)를 로드합니다.
    
    파일명 예시: 20250501_Hanwha_vs_LG.json
    
    Args:
        data_dir: 경기 데이터 디렉토리 경로
    
    Returns:
        List[Dict]: 로드된 경기 데이터 목록
    """
    if data_dir is None:
        data_dir = MATCH_DATA_DIR
    
    match_data = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️ 경기 데이터 디렉토리가 존재하지 않습니다: {data_path}")
        return match_data
    
    json_files = list(data_path.glob("*.json"))
    print(f"📂 경기 데이터 파일 발견: {len(json_files)}개")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 메타데이터 추가
            data["_source_file"] = json_file.name
            data["_data_type"] = "match"
            data["_loaded_at"] = datetime.now().isoformat()
            
            # 파일명에서 날짜/팀 정보 추출 (예: 20250501_Hanwha_vs_LG)
            filename = json_file.stem
            
            # 날짜 추출
            date_match = re.search(r'(\d{8})', filename)
            if date_match:
                date_str = date_match.group(1)
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                data["date"] = formatted_date
            
            # 팀 추출 (vs 또는 _ 구분)
            team_pattern = r'([A-Za-z]+)(?:_vs_|vs|_)([A-Za-z]+)'
            team_match = re.search(team_pattern, filename, re.IGNORECASE)
            if team_match:
                data["teams"] = [team_match.group(1), team_match.group(2)]
            
            match_data.append(data)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
        except Exception as e:
            print(f"❌ 파일 로드 오류 ({json_file.name}): {e}")
    
    return match_data


# =============================================================================
# 문서 변환 함수
# =============================================================================

def create_document_from_data(data: Dict, data_type: str) -> Document:
    """
    JSON 데이터를 LangChain Document로 변환합니다.
    
    핵심 전략 (논문 Section 4.2):
    - page_content: 설명 문장 (임베딩 대상)
    - metadata: 원본 JSON + 필터링용 메타데이터
    
    Args:
        data: JSON 데이터
        data_type: "season" 또는 "match"
    
    Returns:
        Document: LangChain Document 객체
    """
    # 1. 설명 문장 생성 (임베딩 대상)
    descriptive_sentence = generate_descriptive_sentence(data, data_type)
    
    # Instruct 모델용 쿼리 포맷
    # 논문 Section 4.4.1에서 검증된 방식
    embedding_text = (
        f"Instruct: Find the baseball dataset covering the given team(s) and statistics. "
        f"Query: {descriptive_sentence}"
    )
    
    # 2. 메타데이터 구성
    metadata = {
        "type": data_type,
        "source_file": data.get("_source_file", ""),
        "raw_data": json.dumps(data, ensure_ascii=False),  # 원본 JSON 저장
    }
    
    # 시즌 데이터 메타데이터
    if data_type == "season":
        metadata["season"] = data.get("season", "2025")
        metadata["team"] = data.get("team", "")
        metadata["stat_type"] = data.get("stat_type", "")
        # teams 필드: 메타데이터 필터링용
        if data.get("team"):
            metadata["teams"] = [data.get("team")]
    
    # 경기 데이터 메타데이터
    elif data_type == "match":
        metadata["date"] = data.get("date", "")
        metadata["teams"] = data.get("teams", [])
        if metadata["teams"]:
            metadata["home_team"] = metadata["teams"][0] if len(metadata["teams"]) > 0 else ""
            metadata["away_team"] = metadata["teams"][1] if len(metadata["teams"]) > 1 else ""
    
    return Document(
        page_content=embedding_text,
        metadata=metadata
    )


def prepare_documents(season_data: List[Dict], match_data: List[Dict]) -> List[Document]:
    """
    모든 데이터를 Document 객체로 변환합니다.
    
    Args:
        season_data: 시즌 데이터 목록
        match_data: 경기 데이터 목록
    
    Returns:
        List[Document]: 변환된 Document 목록
    """
    documents = []
    
    print("\n🔄 시즌 데이터 변환 중...")
    for data in tqdm(season_data, desc="Season"):
        doc = create_document_from_data(data, "season")
        documents.append(doc)
    
    print("\n🔄 경기 데이터 변환 중...")
    for data in tqdm(match_data, desc="Match"):
        doc = create_document_from_data(data, "match")
        documents.append(doc)
    
    return documents


# =============================================================================
# ChromaDB 적재 함수
# =============================================================================

def initialize_vector_store(
    documents: List[Document] = None,
    persist_directory: str = None,
    collection_name: str = None
) -> Chroma:
    """
    ChromaDB 벡터 스토어를 초기화하거나 기존 스토어를 로드합니다.
    
    Args:
        documents: 적재할 Document 목록 (None이면 기존 스토어 로드)
        persist_directory: 영구 저장 디렉토리
        collection_name: 컬렉션 이름
    
    Returns:
        Chroma: 초기화된 벡터 스토어
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DB_DIR)
    
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    # 임베딩 모델 초기화
    embeddings = get_embedding_model()
    
    if documents:
        # 새로운 문서로 벡터 스토어 생성
        print(f"\n📦 ChromaDB 초기화 중... (문서 수: {len(documents)})")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        print(f"✅ ChromaDB 적재 완료: {persist_directory}")
        
    else:
        # 기존 벡터 스토어 로드
        print(f"\n📂 기존 ChromaDB 로드 중: {persist_directory}")
        
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    
    return vector_store


def clear_vector_store(persist_directory: str = None, collection_name: str = None):
    """
    기존 벡터 스토어를 삭제합니다.
    
    Args:
        persist_directory: 영구 저장 디렉토리
        collection_name: 컬렉션 이름
    """
    if persist_directory is None:
        persist_directory = str(CHROMA_DB_DIR)
    
    if collection_name is None:
        collection_name = COLLECTION_NAME
    
    import shutil
    persist_path = Path(persist_directory)
    
    if persist_path.exists():
        shutil.rmtree(persist_path)
        print(f"🗑️ 기존 벡터 스토어 삭제: {persist_directory}")


# =============================================================================
# 메인 적재 파이프라인
# =============================================================================

def ingest_all_data(
    season_dir: Path = None,
    match_dir: Path = None,
    clear_existing: bool = True
) -> Chroma:
    """
    모든 데이터를 로드하고 ChromaDB에 적재하는 메인 파이프라인입니다.
    
    Args:
        season_dir: 시즌 데이터 디렉토리
        match_dir: 경기 데이터 디렉토리
        clear_existing: 기존 데이터 삭제 여부
    
    Returns:
        Chroma: 초기화된 벡터 스토어
    
    Example:
        >>> from src.ingest import ingest_all_data
        >>> vector_store = ingest_all_data()
        >>> print(f"적재된 문서 수: {vector_store._collection.count()}")
    """
    print("=" * 60)
    print("🚀 KBO 데이터 적재 파이프라인 시작")
    print("=" * 60)
    
    # 1. 기존 데이터 삭제 (선택적)
    if clear_existing:
        clear_vector_store()
    
    # 2. 데이터 로드
    print("\n📥 데이터 로드 중...")
    season_data = load_season_data(season_dir)
    match_data = load_match_data(match_dir)
    
    if not season_data and not match_data:
        print("⚠️ 로드된 데이터가 없습니다. 데이터 디렉토리를 확인하세요.")
        return None
    
    print(f"\n📊 로드된 데이터 요약:")
    print(f"   - 시즌 데이터: {len(season_data)}건")
    print(f"   - 경기 데이터: {len(match_data)}건")
    
    # 3. Document 변환
    documents = prepare_documents(season_data, match_data)
    
    # 4. ChromaDB 적재
    vector_store = initialize_vector_store(documents)
    
    # 5. 결과 확인
    doc_count = vector_store._collection.count()
    print(f"\n✅ 적재 완료! 총 {doc_count}개의 문서가 저장되었습니다.")
    print("=" * 60)
    
    return vector_store


# =============================================================================
# CLI 실행
# =============================================================================

if __name__ == "__main__":
    """
    명령줄에서 직접 실행:
    python -m src.ingest
    """
    vector_store = ingest_all_data()
    
    if vector_store:
        # 간단한 테스트 쿼리
        print("\n🔍 테스트 검색 실행...")
        results = vector_store.similarity_search("한화 이글스 시즌 성적", k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n결과 {i}:")
            print(f"  내용: {doc.page_content[:100]}...")
            print(f"  타입: {doc.metadata.get('type')}")
