"""
Classifier module - 쿼리 분류 및 의도 파악

사용자 쿼리를 분석하여 의도(intent)를 분류하고 엔티티를 추출합니다.
논문 Section 4.3.1 (Query Classification) 전략을 구현합니다.

분류 카테고리:
1. general - 일반 질문 (KPI 설명, 규칙 질문 등)
2. season_analysis - 시즌 분석 (팀 시즌 성적, 순위, 누적 통계)
3. match_analysis - 경기 분석 (특정 경기 결과, 선수 활약)
"""

import json
from typing import Dict, Optional, List, Tuple, Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .config import CLASSIFIER_MODEL, OPENAI_API_KEY, TEMPERATURE
from .utils import extract_teams_from_query, extract_date_from_query, normalize_team_name


# =============================================================================
# 출력 스키마 정의
# =============================================================================

class ClassificationResult(BaseModel):
    """쿼리 분류 결과 스키마"""
    
    reasoning_steps: str = Field(
        description="분류 과정에 대한 단계별 추론 (Chain-of-Thought)"
    )
    query_type: str = Field(
        description="쿼리 유형: 'general', 'season_analysis', 'match_analysis'"
    )
    teams: List[str] = Field(
        default=[],
        description="추출된 팀 목록 (정규화된 영문명)"
    )
    date: Optional[str] = Field(
        default=None,
        description="추출된 날짜 (YYYY-MM-DD 형식)"
    )
    confidence: float = Field(
        default=0.0,
        description="분류 신뢰도 (0.0 ~ 1.0)"
    )


# =============================================================================
# 분류 프롬프트 템플릿
# =============================================================================

CLASSIFICATION_PROMPT = """당신은 KBO 한국 프로야구 데이터 분석 시스템의 쿼리 분류기입니다.
사용자의 질문을 분석하여 의도를 파악하고, 관련 엔티티(팀명, 날짜)를 추출하세요.

## 분류 카테고리

1. **general** (일반 질문)
   - 야구 규칙, 용어 설명 (예: "WAR가 뭐야?", "타율은 어떻게 계산해?")
   - 특정 데이터 조회가 필요 없는 질문
   - 인사, 잡담

2. **season_analysis** (시즌 분석) - 선수의 시즌 성적 조회
   - **선수 개인 시즌 성적, 통계 조회** (핵심!)
   - 예: "김택연 2024 성적", "류현진 시즌 분석", "2025년 원태인 성적"
   - 키워드: 시즌, 올해, 순위, 전체, 누적, 분석, 성적, 통계
   - **주의**: 선수 이름만 언급되어도 성적/분석/통계를 요청하면 season_analysis로 분류

3. **match_analysis** (경기 분석)
   - 특정 경기 결과 또는 예정 경기
   - 예: "어제 한화 LG 경기 어땠어?", "5월 1일 삼성 경기"
   - 키워드: 경기, 어제, 오늘, vs, 대, 맞대결

## KBO 팀 목록 (한글 → 영문 정규화)
- 한화, 이글스 → Hanwha
- LG, 엘지, 트윈스 → LG
- 삼성, 라이온즈 → Samsung
- 두산, 베어스 → Doosan
- 롯데, 자이언츠 → Lotte
- 기아, KIA, 타이거즈 → KIA
- NC, 다이노스 → NC
- SSG, 랜더스 → SSG
- 키움, 히어로즈 → Kiwoom
- KT, 위즈 → KT

## 사용자 쿼리
"{query}"

## 지시사항
1. 먼저 reasoning_steps에서 단계별로 분석하세요 (Chain-of-Thought)
2. query_type을 결정하세요
3. 팀명이 언급되면 영문 정규화명으로 teams에 추가하세요
4. 날짜 정보가 있으면 YYYY-MM-DD 형식으로 date에 기록하세요
5. 분류 신뢰도(0.0~1.0)를 confidence에 기록하세요

## 예시

쿼리: "한화 어제 경기 어땠어?"
→ reasoning_steps: "1. '어제 경기'라는 표현에서 특정 경기를 분석하는 것으로 판단. 2. '한화'팀 언급됨. 3. 경기 분석 유형으로 분류."
→ query_type: "match_analysis"
→ teams: ["Hanwha"]
→ date: (어제 날짜 계산)
→ confidence: 0.95

쿼리: "OPS가 뭐야?"
→ reasoning_steps: "1. 야구 통계 용어에 대한 질문. 2. 특정 팀이나 경기 언급 없음. 3. 일반 질문으로 분류."
→ query_type: "general"
→ teams: []
→ date: null
→ confidence: 0.98

쿼리: "올시즌 LG 타선 분석"
→ reasoning_steps: "1. '올시즌'으로 시즌 전체 분석 요청. 2. 'LG' 팀 언급됨. 3. '타선 분석'은 시즌 누적 통계 필요. 4. 시즌 분석으로 분류."
→ query_type: "season_analysis"
→ teams: ["LG"]
→ date: null
→ confidence: 0.92

쿼리: "2024년 김택연 성적"
→ reasoning_steps: "1. '김택연'은 선수 이름임. 2. '2024년'은 특정 시즌을 의미. 3. '성적'은 시즌 누적 통계 조회 요청. 4. 선수 개인 시즌 성적 조회이므로 season_analysis로 분류."
→ query_type: "season_analysis"
→ teams: []
→ date: null
→ confidence: 0.90

쿼리: "류현진 시즌 분석"
→ reasoning_steps: "1. '류현진'은 선수 이름. 2. '시즌 분석'은 시즌 누적 통계 요청. 3. 선수 개인 시즌 성적 조회이므로 season_analysis로 분류."
→ query_type: "season_analysis"
→ teams: []
→ date: null
→ confidence: 0.92

JSON 형식으로만 응답하세요:
"""


# =============================================================================
# 분류기 클래스
# =============================================================================

class QueryClassifier:
    """
    LLM 기반 쿼리 분류기
    
    Chain-of-Thought 프롬프팅을 사용하여 쿼리 의도를 분류합니다.
    """
    
    def __init__(self, model: str = None, temperature: float = None):
        """
        분류기 초기화
        
        Args:
            model: 사용할 LLM 모델명
            temperature: 모델 온도 (0 = 결정적 출력)
        """
        self.model = model or CLASSIFIER_MODEL
        self.temperature = temperature if temperature is not None else TEMPERATURE
        
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
    
    def classify(self, query: str) -> ClassificationResult:
        """
        사용자 쿼리를 분류합니다.
        
        Args:
            query: 사용자 쿼리
        
        Returns:
            ClassificationResult: 분류 결과
        """
        # 1. LLM 호출
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query})
        
        # 2. JSON 파싱
        try:
            # LLM 응답에서 JSON 추출
            content = response.content
            
            # JSON 블록 추출 (마크다운 코드 블록 처리)
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            result_dict = json.loads(json_str)
            
            # 팀명 정규화 보정 (LLM이 잘못 추출한 경우 대비)
            normalized_teams = []
            for team in result_dict.get("teams", []):
                norm_team, score = normalize_team_name(team)
                if norm_team:
                    normalized_teams.append(norm_team)
                else:
                    normalized_teams.append(team)  # 원본 유지
            
            result_dict["teams"] = normalized_teams
            
            # 날짜 보정
            if not result_dict.get("date"):
                extracted_date = extract_date_from_query(query)
                if extracted_date:
                    result_dict["date"] = extracted_date
            
            return ClassificationResult(**result_dict)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 파싱 실패시 폴백: 규칙 기반 분류
            print(f"⚠️ LLM 응답 파싱 실패, 규칙 기반 분류 수행: {e}")
            return self._fallback_classify(query)
    
    def _fallback_classify(self, query: str) -> ClassificationResult:
        """
        LLM 실패시 규칙 기반 폴백 분류
        
        Args:
            query: 사용자 쿼리
        
        Returns:
            ClassificationResult: 분류 결과
        """
        query_lower = query.lower()
        
        # 팀 추출
        teams = extract_teams_from_query(query)
        team_names = [t[0] for t in teams]
        
        # 날짜 추출
        date = extract_date_from_query(query)
        
        # 분류 규칙
        match_keywords = ["경기", "어제", "오늘", "내일", "vs", "대", "맞대결", "결과"]
        season_keywords = ["시즌", "올해", "순위", "전체", "누적", "분석", "성적", "통계", "년"]
        general_keywords = ["뭐", "무엇", "어떻게", "왜", "설명", "알려"]
        
        # 키워드 매칭
        has_match_keyword = any(kw in query_lower for kw in match_keywords)
        has_season_keyword = any(kw in query_lower for kw in season_keywords)
        has_general_keyword = any(kw in query_lower for kw in general_keywords)
        
        # 분류 결정
        if date or (has_match_keyword and team_names):
            query_type = "match_analysis"
            confidence = 0.7
            reasoning = "날짜 또는 경기 키워드 감지됨 (규칙 기반)"
        elif has_season_keyword and team_names:
            query_type = "season_analysis"
            confidence = 0.7
            reasoning = "시즌 분석 키워드와 팀명 감지됨 (규칙 기반)"
        elif has_season_keyword and not has_general_keyword:
            # 선수 성적 조회 등: 팀 없이 시즌/성적 키워드만 있는 경우
            query_type = "season_analysis"
            confidence = 0.65
            reasoning = "시즌/성적 분석 키워드 감지됨 (선수 개인 성적 조회 추정, 규칙 기반)"
        elif team_names and not has_general_keyword:
            # 팀만 언급된 경우 시즌 분석으로 기본 분류
            query_type = "season_analysis"
            confidence = 0.6
            reasoning = "팀명만 감지됨, 시즌 분석으로 기본 분류 (규칙 기반)"
        else:
            query_type = "general"
            confidence = 0.5
            reasoning = "특정 분석 패턴 미감지 (규칙 기반 폴백)"
        
        return ClassificationResult(
            reasoning_steps=reasoning,
            query_type=query_type,
            teams=team_names,
            date=date,
            confidence=confidence
        )


# =============================================================================
# 싱글톤 인스턴스 및 편의 함수
# =============================================================================

_classifier_instance: Optional[QueryClassifier] = None


def get_classifier() -> QueryClassifier:
    """
    분류기 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        QueryClassifier: 분류기 인스턴스
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance


def classify_query(query: str) -> ClassificationResult:
    """
    쿼리를 분류하는 편의 함수
    
    Args:
        query: 사용자 쿼리
    
    Returns:
        ClassificationResult: 분류 결과
    
    Example:
        >>> from src.classifier import classify_query
        >>> result = classify_query("한화 어제 경기 어땠어?")
        >>> print(result.query_type)
        "match_analysis"
        >>> print(result.teams)
        ["Hanwha"]
    """
    classifier = get_classifier()
    return classifier.classify(query)


# =============================================================================
# 선질문 (Pre-question) 생성 및 사용자 선택 기반 분류
# =============================================================================

def generate_pre_question() -> str:
    """
    사용자에게 질문 유형을 선택하게 하는 선질문을 생성합니다.
    
    제일 처음 사용자에게 질문 유형을 물어보고,
    그 응답에 따라 API 호출 또는 RAG 사용을 결정합니다.
    
    Returns:
        str: 사용자에게 제시할 선질문 텍스트
    
    Example:
        >>> pre_question = generate_pre_question()
        >>> print(pre_question)
        "어떤 유형의 질문이신가요?
        1) 일반 질문
        2) 선수의 시즌 성적
        3) 특정 경기 분석"
    """
    pre_question = """어떤 유형의 질문이신가요?

1️⃣ 일반 질문 (야구 규칙, 용어 설명, 일반 상식)
   예: "OPS가 뭐야?", "WAR는 어떻게 계산해?", "야구 규칙 설명해줄래?"

2️⃣ 선수의 시즌 성적 (특정 선수의 시즌 통계 및 분석)
   예: "김택연 2024 성적", "류현진 올해 성적", "원태인 2025 시즌 분석"

3️⃣ 특정 경기 분석 (경기 결과, 선수 활약, 경기 통계)
   예: "어제 한화 LG 경기 어땠어?", "5월 1일 삼성 경기 결과"

➡️ 답변: 1, 2, 또는 3 입력"""
    
    return pre_question


class PreQuestionChoice:
    """사용자의 선질문 응답을 파싱하는 클래스"""
    
    # 매핑: 사용자 입력 → query_type
    CHOICE_MAPPING = {
        # 숫자
        "1": "general",
        "2": "season_analysis",
        "3": "match_analysis",
        # 한글 약자
        "일반": "general",
        "분석": "season_analysis",
        "경기": "match_analysis",
        # 전체 단어
        "일반질문": "general",
        "시즌분석": "season_analysis",
        "시즌": "season_analysis",
        "경기분석": "match_analysis",
        "경기": "match_analysis",
    }
    
    @staticmethod
    def parse_choice(user_input: str) -> Optional[str]:
        """
        사용자 입력을 파싱하여 query_type을 반환합니다.
        
        Args:
            user_input: 사용자 입력 ("1", "2", "3", "일반", "분석", "경기" 등)
        
        Returns:
            Optional[str]: query_type ("general", "season_analysis", "match_analysis")
                          또는 None (인식 불가능한 입력)
        
        Example:
            >>> PreQuestionChoice.parse_choice("1")
            "general"
            >>> PreQuestionChoice.parse_choice("분석")
            "season_analysis"
            >>> PreQuestionChoice.parse_choice("xyz")
            None
        """
        normalized = user_input.strip().lower()
        return PreQuestionChoice.CHOICE_MAPPING.get(normalized)


def classify_by_user_choice(
    query: str,
    user_choice: str
) -> ClassificationResult:
    """
    사용자의 선질문 응답을 기반으로 쿼리를 분류합니다.
    
    API 호출 없이 사용자 입력으로 직접 분류하므로 빠릅니다.
    
    Args:
        query: 사용자의 원본 질문
        user_choice: 선질문에 대한 사용자의 답변 ("1", "2", "3" 등)
    
    Returns:
        ClassificationResult: 분류 결과
    
    Raises:
        ValueError: 인식 불가능한 선택지인 경우
    
    Example:
        >>> result = classify_by_user_choice("한화 성적", "2")
        >>> print(result.query_type)
        "season_analysis"
        >>> print(result.confidence)
        1.0  # 사용자 선택이므로 신뢰도 100%
    """
    # 사용자 선택 파싱
    query_type = PreQuestionChoice.parse_choice(user_choice)
    
    if query_type is None:
        raise ValueError(
            f"인식 불가능한 선택입니다: '{user_choice}'\n"
            f"1 (일반), 2 (분석), 3 (경기) 중 선택해주세요."
        )
    
    # 팀/날짜 추출 (여전히 필요)
    teams = extract_teams_from_query(query)
    team_names = [t[0] for t in teams]
    date = extract_date_from_query(query) if query_type in ["match_analysis", "season_analysis"] else None
    
    return ClassificationResult(
        reasoning_steps=f"사용자 선택: {user_choice} → {query_type}",
        query_type=query_type,
        teams=team_names,
        date=date,
        confidence=1.0  # 사용자 선택이므로 완벽한 신뢰도
    )


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    # 테스트 쿼리
    test_queries = [
        "한화 어제 경기 어땠어?",
        "OPS가 뭐야?",
        "올시즌 LG 타선 분석해줘",
        "5월 1일 삼성 롯데 경기 결과",
        "두산 순위가 어떻게 돼?",
        "ERA 계산 방법 알려줘",
        "한화 vs LG 맞대결 기록",
    ]
    
    classifier = QueryClassifier()
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"쿼리: {query}")
        print(f"{'='*50}")
        
        result = classifier.classify(query)
        print(f"분류: {result.query_type}")
        print(f"팀: {result.teams}")
        print(f"날짜: {result.date}")
        print(f"신뢰도: {result.confidence:.2f}")
        print(f"추론: {result.reasoning_steps}")
