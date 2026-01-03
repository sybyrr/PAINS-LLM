"""
KBO 야구 분석 챗봇 - 메인 진입점

사용법:
    # 대화형 CLI 실행
    python main.py
    
    # 데이터 적재 실행
    python main.py --ingest
    
    # 단일 쿼리 실행
    python main.py --query "한화 올시즌 성적 알려줘"
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="KBO 야구 분석 챗봇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py              # 대화형 모드 실행
  python main.py --ingest     # 데이터 적재
  python main.py -q "한화 성적" # 단일 질문
        """
    )
    
    parser.add_argument(
        "--ingest", "-i",
        action="store_true",
        help="데이터를 벡터 스토어에 적재"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="단일 질문 실행"
    )
    
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="벡터 DB 초기화 후 재적재"
    )
    
    args = parser.parse_args()
    
    # 데이터 적재
    if args.ingest or args.reset_db:
        print("📦 데이터 적재를 시작합니다...")
        from src.ingest import ingest_all_data
        
        if args.reset_db:
            import shutil
            from src.config import CHROMA_PATH
            if Path(CHROMA_PATH).exists():
                print(f"🗑️ 기존 DB 삭제: {CHROMA_PATH}")
                shutil.rmtree(CHROMA_PATH)
        
        result = ingest_all_data()
        print(f"✅ 적재 완료: {result['total_documents']} 문서")
        return
    
    # 단일 쿼리
    if args.query:
        print(f"🔍 질문: {args.query}")
        from src.agent import chat
        response = chat(args.query)
        
        # 검색 정보 출력
        print(f"\n📑 검색 정보:")
        print(f"   - 유사도: {response.retrieval_score:.2%}")
        print(f"   - 검색 방법: {response.retrieval_method}")
        if response.retrieved_doc_info:
            doc = response.retrieved_doc_info
            print(f"   - 문서 타입: {doc.get('type')}")
            print(f"   - 팀: {doc.get('teams')}")
            if doc.get('date'):
                print(f"   - 날짜: {doc.get('date')}")
            if doc.get('player_name'):
                print(f"   - 선수: {doc.get('player_name')}")
        
        print(f"\n🤖 답변:\n{response.response}")
        
        if response.dashboard:
            print(f"\n📊 대시보드 생성됨 (위젯 {len(response.dashboard.get('widgets', []))}개)")
        return
    
    # 대화형 모드 (기본)
    from src.agent import run_interactive_chat
    run_interactive_chat()


if __name__ == "__main__":
    main()
