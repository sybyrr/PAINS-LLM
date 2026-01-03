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
from langchain_core.documents import Document
from tqdm import tqdm

from .config import (
    CHROMA_DB_DIR, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL,
    SEASON_DATA_DIR,
    MATCH_DATA_DIR,
    PROCESSED_DATA_DIR
)

from .utils import generate_game_description, generate_descriptive_sentence, TEAM_MAP

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
    
    현재 JSON 구조:
    [
      {
        "dataset_id": "2025_REGULAR_PITCHING_STATS",
        "name": "2025 Regular league Pitching Stats",
        "type": "player",
        "headers": [...],
        "data": [
          { "Rank": 1, "Name": "...", "Team": "...", "ERA": ..., ... },
          ...
        ]
      }
    ]
    
    Args:
        data_dir: 시즌 데이터 디렉토리 경로
    
    Returns:
        List[Dict]: 로드된 개별 선수 시즌 데이터 목록
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
                file_content = json.load(f)
            
            # 파일명에서 시즌 타입 추출 (예: 2025_POST_PITCHING_STATS)
            filename = json_file.stem
            is_postseason = "POST" in filename.upper()
            season_type = "Post" if is_postseason else "Regular"
            
            # 연도 추출
            year_match = re.search(r'(\d{4})', filename)
            year = year_match.group(1) if year_match else "2025"
            
            # 통계 타입 추출 (PITCHING, BATTING 등)
            stat_type = "pitching" if "PITCHING" in filename.upper() else "batting"
            
            # 배열 내 각 데이터셋 처리
            if isinstance(file_content, list):
                for dataset in file_content:
                    dataset_name = dataset.get("name", "")
                    dataset_id = dataset.get("dataset_id", "")
                    
                    # data 배열 내 각 레코드를 개별 문서로 처리
                    records = dataset.get("data", [])
                    print(f"   - {json_file.name}: {len(records)}개 선수 레코드 발견")
                    
                    for record in records:
                        # 개별 레코드에 메타데이터 추가
                        record["_source_file"] = json_file.name
                        record["_data_type"] = "season"
                        record["_loaded_at"] = datetime.now().isoformat()
                        record["_dataset_name"] = dataset_name
                        record["_dataset_id"] = dataset_id
                        record["_season_type"] = season_type
                        record["_stat_type"] = stat_type
                        record["season"] = year
                        
                        # Team 필드가 있으면 team으로도 저장
                        if "Team" in record:
                            record["team"] = record["Team"]
                            record["teams"] = [record["Team"]]
                        
                        season_data.append(record)
            else:
                # 단일 객체인 경우 기존 로직 유지
                file_content["_source_file"] = json_file.name
                file_content["_data_type"] = "season"
                file_content["_loaded_at"] = datetime.now().isoformat()
                
                season_match = re.search(r'(\d{4})', filename)
                if season_match:
                    file_content["season"] = season_match.group(1)
                
                season_data.append(file_content)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
        except Exception as e:
            print(f"❌ 파일 로드 오류 ({json_file.name}): {e}")
    
    return season_data


def load_match_data(data_dir: Path = None) -> List[Dict]:
    """
    경기별 투수/타자 기록 데이터를 로드합니다.
    
    같은 날짜 + 같은 팀의 기록들을 하나의 문서로 그룹화합니다.
    예: 2025-10-06 NC팀의 모든 투수 기록 → 하나의 문서
    
    현재 JSON 구조:
    [
      {
        "dataset_id": "2025_POST_MATCH_PITCHING_STATS",
        "name": "2025 Post Match Pitching Data",
        "headers": [...],
        "data": [
          { "Season": 2025, "Date": "2025-10-06", "Team": "NC", "Name": "구창모", 
            "IP": 22, "ER": 1, "SO": 0, "Result": "승", ... },
          ...
        ]
      }
    ]
    
    Args:
        data_dir: 경기 데이터 디렉토리 경로
    
    Returns:
        List[Dict]: 날짜+팀으로 그룹화된 경기 기록 목록
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
                file_content = json.load(f)
            
            # 파일명에서 시즌 타입 및 기록 타입 추출
            filename = json_file.stem.upper()
            is_postseason = "POST" in filename
            season_type = "Post" if is_postseason else "Regular"
            
            # 투수/타자 기록 타입 추출
            is_pitching = "PITCHING" in filename
            record_type = "pitcher" if is_pitching else "batter"
            
            # 연도 추출
            year_match = re.search(r'(\d{4})', filename)
            year = year_match.group(1) if year_match else "2025"
            
            # 배열 내 각 데이터셋 처리
            if isinstance(file_content, list):
                for dataset in file_content:
                    dataset_name = dataset.get("name", "")
                    dataset_id = dataset.get("dataset_id", "")
                    
                    # data 배열 내 각 레코드를 날짜+팀으로 그룹화
                    records = dataset.get("data", [])
                    print(f"   - {json_file.name}: {len(records)}개 레코드 발견")
                    
                    # 날짜+팀 기준으로 그룹화
                    grouped = {}
                    for record in records:
                        date = record.get("Date", "Unknown")
                        team = record.get("Team", "Unknown")
                        key = (date, team)
                        
                        if key not in grouped:
                            grouped[key] = []
                        grouped[key].append(record)
                    
                    print(f"     → {len(grouped)}개 경기(날짜+팀)로 그룹화")
                    
                    # 그룹화된 데이터를 문서로 변환
                    for (date, team), player_records in grouped.items():
                        grouped_doc = {
                            "_source_file": json_file.name,
                            "_data_type": "match",
                            "_loaded_at": datetime.now().isoformat(),
                            "_dataset_name": dataset_name,
                            "_dataset_id": dataset_id,
                            "_season_type": season_type,
                            "_record_type": record_type,
                            "_year": year,
                            "Date": date,
                            "date": date,
                            "Team": team,
                            "teams": [team],
                            "players": player_records  # 해당 경기의 모든 선수 기록
                        }
                        match_data.append(grouped_doc)
            else:
                print(f"⚠️ 예상치 못한 JSON 구조: {json_file.name}")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
        except Exception as e:
            print(f"❌ 파일 로드 오류 ({json_file.name}): {e}")
    
    return match_data


# === 전처리된 경기별 데이터 로드 함수 ===

def load_processed_game_data(data_dir: Path = None) -> List[Dict]:
    """
    전처리된 경기별 JSON 데이터를 로드합니다.
    
    preprocess.ipynb에서 생성한 data/processed/matches/ 폴더의 JSON 파일을 읽어옵니다.
    각 JSON 파일은 경기별로 양 팀 투수 기록이 통합되어 있습니다.
    
    Args:
        data_dir: 전처리된 데이터 디렉토리 경로 (None일 경우 PROCESSED_DATA_DIR 사용)
    
    Returns:
        List[Dict]: 경기별 데이터 리스트
    """
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR
    
    game_data = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️ 전처리된 데이터 디렉토리가 존재하지 않습니다: {data_path}")
        print("   먼저 preprocess.ipynb를 실행하여 데이터를 전처리하세요.")
        return game_data
    
    json_files = list(data_path.glob("*.json"))
    print(f"📂 전처리된 데이터 파일 발견: {len(json_files)}개")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            # JSON 구조: [ { "dataset_id": ..., "data": [...] }, ... ]
            if isinstance(content, list):
                for dataset in content:
                    games = dataset.get('data', [])
                    print(f"   - {json_file.name}: {len(games)}개 경기")
                    game_data.extend(games)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류 ({json_file.name}): {e}")
        except Exception as e:
            print(f"❌ 파일 로드 오류 ({json_file.name}): {e}")
    
    return game_data


def prepare_game_documents(game_data: List[Dict]) -> List[Document]:
    """
    전처리된 경기별 데이터를 LangChain Document로 변환합니다.
    
    각 경기(양 팀 투수 기록 통합)에 대해 하나의 Document를 생성합니다.
    
    Args:
        game_data: 경기별 데이터 리스트
    
    Returns:
        List[Document]: LangChain Document 리스트
    """
    documents = []
    
    print(f"🔄 {len(game_data)}개의 경기 데이터를 Document로 변환 중...")
    
    for game in game_data:
        # 1. 설명 문장 생성 (임베딩 대상)
        description = generate_game_description(game)
        
        # 2. 메타데이터 구성
        metadata = {
            "type": "game",
            "game_id": game.get("game_id", ""),
            "date": game.get("date", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "home_runs": game.get("home_runs", ""),
            "away_runs": game.get("away_runs", ""),
            "season_type": game.get("season_type", "Regular"),
            "home_pitcher_count": game.get("home_pitcher_count", 0),
            "away_pitcher_count": game.get("away_pitcher_count", 0),
            "total_pitcher_count": game.get("total_pitcher_count", 0),
            # 원본 데이터 저장 (RAG 검색 후 복원용)
            "original_data": json.dumps(game, ensure_ascii=False)
        }
        
        doc = Document(page_content=description, metadata=metadata)
        documents.append(doc)
    
    return documents


def prepare_season_documents(season_data: List[Dict]) -> List[Document]:
    """
    시즌 데이터를 LangChain Document로 변환합니다.
    
    Args:
        season_data: 시즌 데이터 리스트
    
    Returns:
        List[Document]: LangChain Document 리스트
    """
    documents = []
    
    print(f"🔄 {len(season_data)}개의 시즌 데이터를 Document로 변환 중...")
    
    for record in season_data:
        # 1. 설명 문장 생성
        description = generate_descriptive_sentence(record, "season")
        
        # 2. 메타데이터 구성
        metadata = {
            "type": "season",
            "season": str(record.get("season", "2025")),
            "season_type": record.get("_season_type", "Regular"),
            "team": record.get("Team", record.get("team", "")),
            "teams": record.get("Team", record.get("team", "")), # For consistency with match data
            "name": record.get("Name", ""),
            "stat_type": record.get("_stat_type", "pitching"),
            # 원본 데이터 저장
            "original_data": json.dumps(record, ensure_ascii=False)
        }
        
        doc = Document(page_content=description, metadata=metadata)
        documents.append(doc)
        
    return documents


# =============================================================================
# ChromaDB 적재 함수
# =============================================================================

def initialize_vector_store(
    documents: List[Document] = None,
    persist_directory: str = None,
    collection_name: str = None,
    batch_size: int = 100
) -> Chroma:
    """
    ChromaDB 벡터 스토어를 초기화하거나 기존 스토어를 로드합니다.
    
    성능 최적화:
    - 배치 단위로 문서를 처리하여 메모리 효율성 향상
    - tqdm으로 진행 상황 표시
    
    Args:
        documents: 적재할 Document 목록 (None이면 기존 스토어 로드)
        persist_directory: 영구 저장 디렉토리
        collection_name: 컬렉션 이름
        batch_size: 한 번에 처리할 문서 수 (기본값: 100)
    
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
        print(f"   배치 크기: {batch_size}개씩 처리")
        
        # 첫 번째 배치로 벡터 스토어 생성
        first_batch = documents[:batch_size]
        vector_store = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # 나머지 배치를 순차적으로 추가
        remaining_docs = documents[batch_size:]
        if remaining_docs:
            total_batches = (len(remaining_docs) + batch_size - 1) // batch_size
            print(f"\n🔄 임베딩 생성 및 적재 중... ({total_batches}개 배치)")
            
            for i in tqdm(range(0, len(remaining_docs), batch_size), desc="Batches"):
                batch = remaining_docs[i:i + batch_size]
                vector_store.add_documents(batch)
        
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
    processed_dir: Path = None,
    season_dir: Path = None,
    clear_existing: bool = True
) -> Chroma:
    """
    전처리된 경기별 데이터와 시즌 데이터를 로드하고 ChromaDB에 적재하는 메인 파이프라인입니다.
    
    Args:
        processed_dir: 전처리된 데이터 디렉토리 (None일 경우 config 기본값 사용)
        season_dir: 시즌 데이터 디렉토리 (None일 경우 config 기본값 사용)
        clear_existing: 기존 데이터 삭제 여부 (기본값 True)
    
    Returns:
        Chroma: 초기화된 벡터 스토어
    
    Note:
        이 함수 실행 전에 preprocess.ipynb를 먼저 실행하여 
        data/processed/matches/ 폴더에 전처리된 JSON 파일을 생성해야 합니다.
    """
    print("=" * 60)
    print("🚀 KBO 데이터 적재 파이프라인 시작")
    print("=" * 60)
    
    # 1. 기존 데이터 삭제 (선택적)
    if clear_existing:
        clear_vector_store()
    
    # 2. 전처리된 경기별 데이터 로드
    print("\n📥 전처리된 경기 데이터 로드 중...")
    game_data = load_processed_game_data(processed_dir)
    
    # 3. 시즌 데이터 로드
    print("\n📥 시즌 데이터 로드 중...")
    season_data = load_season_data(season_dir)
    
    if not game_data and not season_data:
        print("⚠️ 데이터가 없습니다.")
        print("   먼저 preprocess.ipynb를 실행하여 데이터를 전처리하거나 시즌 데이터를 확인하세요.")
        return None
    
    print(f"\n📊 로드된 데이터 요약:")
    print(f"   - 경기 수: {len(game_data)}")
    print(f"   - 시즌 데이터 레코드: {len(season_data)}")
    
    # 시즌별 통계
    regular_games = [g for g in game_data if g.get('season_type') == 'Regular']
    post_games = [g for g in game_data if g.get('season_type') == 'Post']
    print(f"   - 정규시즌: {len(regular_games)}경기")
    print(f"   - 포스트시즌: {len(post_games)}경기")
    
    # 4. Document 변환
    documents = []
    
    if game_data:
        documents.extend(prepare_game_documents(game_data))
        
    if season_data:
        documents.extend(prepare_season_documents(season_data))
    
    if not documents:
        print("⚠️ 생성된 문서가 없습니다. 데이터 무결성을 확인하세요.")
        return None

    # 5. ChromaDB 적재
    vector_store = initialize_vector_store(documents)
    
    # 6. 결과 확인
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
