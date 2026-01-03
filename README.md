# PAINS-LLM

**KBO 한국 프로야구 특화 LLM 챗봇**

논문 "A Chatbot for Football Analytics: A Deep Dive into RAG, LLM Orchestration and Function Calling" 아키텍처를 KBO 야구에 적용한 RAG 기반 분석 챗봇입니다.

This project is based on 
Blomgren, N. (2025). A Chatbot for Football Analytics : A deep dive into RAG, LLM Orchestration and Function Calling (Dissertation). Retrieved from https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-368039


---

## architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              사용자 쿼리                                      │
│                   "2025년 6월 15일 롯데와 SSG 경기 분석해줘"                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1️⃣ Query Classification                             │
│                     (classifier.py - gpt-4o-mini)                           │
│  ┌─────────────────┬─────────────────┬─────────────────────────────────┐    │
│  │    general      │ season_analysis │      match_analysis             │    │
│  │  (일반 질문)    │   (시즌 분석)   │       (경기 분석)               │    │
│  └────────┬────────┴────────┬────────┴──────────────┬──────────────────┘    │
└───────────│─────────────────│───────────────────────│───────────────────────┘
            │                 │                       │
            ▼                 ▼                       ▼
    ┌───────────────┐  ┌─────────────────────────────────────┐
    │ LLM 직접 응답 │  │  2️⃣ Query Normalization (utils.py)   │
    │ (야구 지식)   │  │  - 팀명 표준화 (RapidFuzz 퍼지매칭)  │
    └───────────────┘  │  - 날짜 추출 및 정규화               │
                       └─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    3️⃣ Hybrid Retrieval (retriever.py)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    ChromaDB Vector Store                            │    │
│  │    • 임베딩: multilingual-e5-large-instruct (1024 dim)             │    │
│  │    • 메타데이터 필터: type, home_team, away_team, date              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────┐     ┌──────────────────────┐                     │
│  │  Semantic Search     │     │   BM25 (Fallback)    │                     │
│  │  (가중치: 0.8)       │     │   (가중치: 0.2)       │                     │
│  └──────────────────────┘     └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              4️⃣ LLM Orchestration (chain.py - gpt-4o)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  LLM-as-Judge: 검색된 데이터가 쿼리와 일치하는지 검증               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  응답 생성: 데이터 기반 분석 + 마크다운 테이블 포맷팅               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                5️⃣ Dashboard Generation (tools.py)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  JSON 대시보드 생성 (프론트엔드 렌더링용)                           │    │
│  │  + matplotlib 시각화 (plt.show() 별도 창)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
PAINS-LLM/
├── main.py                 # 실제 LLM 테스트용 파일
├── main.ipynb              # 개발 및 테스트용 파일 (데이터베이스 구축 + @)
├── requirements.txt        # 패키지 설치
├── .env                    # API 키 설정 (OPENAI_API_KEY)
│
├── src/                    
│   ├── __init__.py
│   ├── config.py           # 환경 변수 및 설정
│   ├── classifier.py       # 쿼리 분류 (general/season/match)
│   ├── utils.py            # 팀명 정규화, 날짜 추출 등 함수들
│   ├── ingest.py           # JSON → ChromaDB 적재
│   ├── retriever.py        # 하이브리드 검색 (Semantic + BM25)
│   ├── chain.py            # 전체 파이프라인 + LLM 응답 생성
│   ├── agent.py            # 대화형 에이전트 + Function Calling
│   └── tools.py            # 대시보드 생성 도구
│
└── data/                   # 데이터 저장소
    ├── chroma_db/          # ChromaDB 벡터 저장소
    ├── gamelog/            # 경기 일정 CSV
    └── raw/                # 원본 JSON 데이터
        ├── matches/        # 경기별 투수 기록
        └── seasons/        # 시즌별 투수 기록
```

---

## detail about `src/`

| 파일 | 역할 | 논문 섹션 |
|------|------|----------|
| `config.py` | 환경 변수, API 키, 경로, 모델 설정 | - |
| `classifier.py` | CoT 프롬프팅으로 쿼리를 general/season/match로 분류 | Section 4.3.1 |
| `utils.py` | RapidFuzz 매칭으로 팀명/선수명 정규화, 날짜 추출 | Section 4.3.2 |
| `ingest.py` | JSON 데이터를 서술형 문장(description)으로 변환 후 ChromaDB에 임베딩 | Section 4.2 |
| `retriever.py` | Semantic + BM25 하이브리드 검색, 메타데이터 필터링 | Section 4.4 |
| `chain.py` | 전체 파이프라인 관리, LLM 응답 생성, 시각화 | Section 4.5 |
| `agent.py` | 대화 메모리, Function Calling, 대화형 CLI | Section 4.5.4 |
| `tools.py` | 대시보드 JSON 생성 도구 (프론트엔드용) | Section 4.5.4 |

---

## How To Use

### 1. 환경 설정

```bash
# 가상 환경 생성
python -m venv .venv

# 활성화 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일 생성: (직접 입력 필요)
```
OPENAI_API_KEY=sk-your-api-key-here 
```

### 3. 데이터 적재

```bash
# 최초 실행 시 또는 데이터 변경 시
python main.py --ingest

# 기존 DB 삭제 후 재적재
python main.py --reset-db
```

### 4. 챗봇 실행

```bash
# 대화형 모드 (기본)
python main.py

# 단일 질문
python main.py -q "2025년 6월 15일 롯데 SSG 경기에서 양 팀 투수들의 기록을 분석해줘"
```

### 5. 대화형 모드 명령어

| 명령어 | 설명 |
|--------|------|
| `/quit` | 종료 |
| `/reset` | 대화 초기화 |
| `/history` | 대화 기록 보기 |
| `/plot` | 마지막 분석 결과 시각화 (matplotlib 창) |

**시각화 키워드**: 질문에 `시각화`, `plot`, `차트`, `그래프`를 포함하면 자동으로 시각화가 표시됩니다.

```
👤 You: 2025년 6월 15일 롯데 SSG 경기 분석해줘 시각화
🤖 Assistant: (분석 결과)
📊 시각화 창이 표시됩니다!
```

---

## 논문 대비 구현 차이점

### 논문과 동일하게 구현한 부분

| 논문 섹션 | 구현 내용 |
|----------|----------|
| Section 4.2 (Embedding) | `multilingual-e5-large-instruct` 모델 사용, L2 정규화 |
| Section 4.3.1 (Query Classification) | CoT 프롬프팅으로 general/season/match 분류 |
| Section 4.3.2 (Query Normalization) | RapidFuzz QRatio scorer로 팀명 매칭 |
| Section 4.4.1 (Instruct Format) | `"Instruct: ... Query: ..."` 포맷 사용 |
| Section 4.4.2 (Hybrid Retrieval) | Semantic 먼저 수행, 이후에 Semantic (0.8) + BM25 (0.2) 앙상블 |
| Section 4.5.2 (LLM-as-Judge) | 검색 결과가 쿼리와 일치하는지 검증 |
| Section 4.5.4 (Function Calling) | 대시보드 JSON 생성 도구 |

### 논문과 다르게 (KBO에 맞게) 수정한 부분

| 항목 | 논문 (축구) | 이 프로젝트 (야구) |
|-----|-----------|------------------|
| **도메인** | 유럽 축구 리그 | KBO 한국 프로야구 |
| **TEAM_MAP** | 축구팀 + 대회 매핑 | 10개 KBO 팀 |
| **데이터 구조** | 리그/경기 JSON | 시즌/경기 JSON (야구 스탯) |
| **Instruct 프롬프트** | "football dataset" | "baseball dataset" |
| **시각화** | PlaymakerAI 사이트 활용 | matplotlib + JSON 대시보드 |

---

## ChromaDB 벡터 저장소 구성

```
data/
└── chroma_db/           ← 로컬 영구 저장소
    ├── chroma.sqlite3   ← 메타데이터 저장
    └── [UUID 폴더들]/   ← 벡터 임베딩 저장
```

| 설정 | 값 |
|------|-----|
| 저장 위치 | `data/chroma_db/` |
| 컬렉션 이름 | `kbo_data` |
| 임베딩 차원 | 1024 |
| 인덱스 방식 | HNSW (ChromaDB 기본값) |
| 유사도 측정 | Cosine Similarity |

> 논문에서는 PlaymakerAI API에서 데이터를 가져왔지만, 이 프로젝트는 로컬 JSON 파일을 사용합니다.

---

## 지원하는 쿼리 유형

### 1. General (일반 질문)
```
"WAR가 뭐야?"
"야구 규칙 설명해줘"
"ERA 계산 방법은?"
```

### 2. Season Analysis (시즌 분석)
```
"한화 올시즌 성적 어때?"
"류현진 2025시즌 분석해줘"
"LG 투수진 성적은?"
```

### 3. Match Analysis (경기 분석)
```
"2025년 6월 15일 롯데와 SSG 경기에서 양 팀 투수들 분석해줘"
"어제 한화 LG 경기에서 투수들 어땠어?"
"10월 31일 포스트시즌 경기 결과"
```

---

## TODO

시각화를 지원하는 쿼리를 확장해야 한다. 지금은 match에서 두 팀의 투수 기록을 묻는 쿼리만 시각화 제공
