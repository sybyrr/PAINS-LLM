"""
Chain module - 정규화 + 라우팅 + 검색 통합 체인

사용자 쿼리 처리의 전체 파이프라인을 관리합니다.
논문 Section 4.3 (User Query Processing) 및 Section 4.5 (LLM Orchestration)를 구현합니다.

파이프라인:
1. 쿼리 정규화 (팀명/선수명 표준화)
2. 의도 분류 (General, Season, Match)
3. 조건부 검색 (분류 결과에 따른 RAG)
4. LLM 응답 생성
"""

import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from .config import LLM_MODEL, OPENAI_API_KEY, TEMPERATURE
from .classifier import classify_query, ClassificationResult
from .retriever import retrieve_for_query
from .utils import extract_teams_from_query, normalize_team_name


# =============================================================================
# 응답 스키마
# =============================================================================

@dataclass
class ChainResult:
    """체인 실행 결과"""
    query: str                          # 원본 쿼리
    query_type: str                     # 분류된 쿼리 유형
    teams: list                         # 추출된 팀 목록
    context: Optional[Dict]             # 검색된 컨텍스트 (있는 경우)
    retrieval_score: float              # 검색 신뢰도
    retrieval_method: str               # 사용된 검색 방법
    response: str                       # LLM 생성 응답
    needs_dashboard: bool               # 대시보드 생성 필요 여부
    validation_passed: bool             # LLM-as-Judge 검증 통과 여부


# =============================================================================
# 시스템 프롬프트 정의
# =============================================================================

SYSTEM_PROMPT_GENERAL = """당신은 KBO 한국 프로야구 데이터 분석 전문가입니다.

역할:
- 야구 통계, 규칙, 용어에 대한 질문에 답변
- 분석 방법론 설명
- 일반적인 야구 지식 제공

지침:
- 정확하고 신뢰할 수 있는 정보만 제공하세요
- 불확실한 경우 명시하세요
- 한국어로 친근하게 답변하세요
"""

SYSTEM_PROMPT_ANALYSIS = """당신은 KBO 한국 프로야구 데이터 분석 전문가입니다.

역할:
- 제공된 데이터를 기반으로 정확한 분석 제공
- 통계적 인사이트 도출
- 팀/선수 성과 평가

## 중요 지침 (LLM-as-Judge)

1. **데이터 검증 우선**: 분석 전 반드시 제공된 데이터가 요청과 일치하는지 확인하세요.
   - 요청된 팀과 데이터의 팀이 일치하는가?
   - 요청된 기간(시즌/경기 날짜)이 데이터와 일치하는가?

2. **불일치 처리**:
   - 완전 일치: 분석 진행
   - 부분 일치 (팀은 맞지만 기간 불일치): 사용자에게 알리고 가용 데이터로 분석
   - 팀 불일치: 분석 거부, 올바른 데이터 요청 안내

3. **응답 형식**:
   - 마크다운 표를 활용한 명확한 데이터 표시
   - 핵심 인사이트를 먼저, 상세 분석은 뒤에
   - 수치는 반드시 데이터 기반으로

4. **시각화 필요성 판단**:
   - 데이터가 풍부하고 비교 분석이 필요하면 대시보드 추천
   - 단순 질문이나 데이터 부족시 텍스트 답변으로 충분
"""

DATA_VALIDATION_PROMPT = """## 데이터 검증

사용자 요청:
- 팀: {requested_teams}
- 유형: {query_type}
- 날짜: {requested_date}

제공된 데이터:
- 데이터 팀: {data_teams}
- 데이터 유형: {data_type}
- 데이터 날짜/시즌: {data_period}

검증 결과를 판단하고, 일치하지 않으면 사용자에게 명확히 알려주세요.
"""


# =============================================================================
# LLM 응답 생성
# =============================================================================

class KBOAnalysisChain:
    """
    KBO 분석 체인
    
    정규화 → 분류 → 검색 → 응답 생성의 전체 파이프라인을 관리합니다.
    """
    
    def __init__(self, model: str = None, temperature: float = None):
        """
        체인 초기화
        
        Args:
            model: 사용할 LLM 모델
            temperature: 모델 온도
        """
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else TEMPERATURE
        
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=OPENAI_API_KEY
        )
    
    def _normalize_query(self, query: str) -> Tuple[str, list]:
        """
        쿼리에서 팀명을 정규화합니다.
        
        Args:
            query: 원본 쿼리
        
        Returns:
            Tuple[str, list]: (정규화된 쿼리, 추출된 팀 목록)
        """
        teams = extract_teams_from_query(query)
        normalized_teams = [t[0] for t in teams]  # (팀명, 점수)에서 팀명만
        
        return query, normalized_teams
    
    def _validate_data_match(
        self,
        classification: ClassificationResult,
        context: Dict
    ) -> Tuple[bool, str]:
        """
        검색된 데이터가 요청과 일치하는지 검증합니다.
        
        논문 Section 4.5.2 (LLM-as-a-judge) 구현
        
        Args:
            classification: 분류 결과
            context: 검색된 컨텍스트
        
        Returns:
            Tuple[bool, str]: (검증 통과 여부, 검증 메시지)
        """
        if not context:
            return False, "검색된 데이터가 없습니다."
        
        requested_teams = set(classification.teams)
        data_teams = set(context.get("teams", []))
        
        # 팀 일치 검사
        if requested_teams and data_teams:
            team_overlap = requested_teams & data_teams
            if not team_overlap:
                return False, f"요청하신 팀({', '.join(requested_teams)})의 데이터를 찾을 수 없습니다. 검색된 데이터: {', '.join(data_teams)}"
        
        # 부분 일치 (경고와 함께 진행)
        if requested_teams and data_teams and requested_teams != team_overlap:
            missing = requested_teams - team_overlap
            return True, f"일부 팀({', '.join(missing)}) 데이터를 찾을 수 없습니다. 가용한 데이터로 분석합니다."
        
        return True, "데이터 검증 완료"
    
    def _generate_response(
        self,
        query: str,
        query_type: str,
        context: Optional[Dict],
        validation_message: str
    ) -> Tuple[str, bool]:
        """
        LLM을 사용하여 응답을 생성합니다.
        
        Args:
            query: 사용자 쿼리
            query_type: 쿼리 유형
            context: 검색된 컨텍스트
            validation_message: 데이터 검증 메시지
        
        Returns:
            Tuple[str, bool]: (응답 텍스트, 대시보드 필요 여부)
        """
        # 시스템 프롬프트 선택
        if query_type == "general":
            system_prompt = SYSTEM_PROMPT_GENERAL
        else:
            system_prompt = SYSTEM_PROMPT_ANALYSIS
        
        # 메시지 구성
        messages = [SystemMessage(content=system_prompt)]
        
        # 컨텍스트가 있는 경우 데이터 포함
        if context and query_type != "general":
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            
            user_content = f"""## 사용자 질문
{query}

## 데이터 검증 상태
{validation_message}

## 분석 데이터
```json
{context_str}
```

위 데이터를 기반으로 분석해주세요. 
데이터가 요청과 일치하지 않으면 그 사실을 먼저 알려주세요.
대시보드(시각화)가 유용할 것 같으면 마지막에 "[대시보드 추천]"을 포함해주세요.
"""
        else:
            user_content = f"""## 사용자 질문
{query}

일반적인 야구 지식을 바탕으로 답변해주세요.
"""
        
        messages.append(HumanMessage(content=user_content))
        
        # LLM 호출
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # 대시보드 필요 여부 판단
        needs_dashboard = "[대시보드 추천]" in response_text
        
        # 대시보드 태그 제거
        response_text = response_text.replace("[대시보드 추천]", "").strip()
        
        return response_text, needs_dashboard
    
    def run(self, query: str) -> ChainResult:
        """
        전체 체인을 실행합니다.
        
        Args:
            query: 사용자 쿼리
        
        Returns:
            ChainResult: 체인 실행 결과
        """
        print(f"\n{'='*60}")
        print(f"🎯 쿼리: {query}")
        print(f"{'='*60}")
        
        # 1. 정규화
        _, normalized_teams = self._normalize_query(query)
        print(f"📝 정규화된 팀: {normalized_teams}")
        
        # 2. 분류
        classification = classify_query(query)
        print(f"🏷️ 분류: {classification.query_type} (신뢰도: {classification.confidence:.2f})")
        print(f"📅 날짜: {classification.date}")
        
        # 팀 정보 병합 (분류기 + 정규화)
        all_teams = list(set(normalized_teams + classification.teams))
        
        # 3. 검색 (분석 쿼리인 경우만)
        context = None
        retrieval_score = 0.0
        retrieval_method = "none"
        validation_passed = True
        validation_message = ""
        
        if classification.query_type != "general":
            context, retrieval_score, retrieval_method = retrieve_for_query(
                query=query,
                query_type=classification.query_type,
                teams=all_teams,
                date=classification.date
            )
            
            print(f"🔍 검색 결과: {retrieval_method} (점수: {retrieval_score:.4f})")
            
            # 4. 데이터 검증
            validation_passed, validation_message = self._validate_data_match(
                classification, context
            )
            print(f"✅ 검증: {validation_message}")
        
        # 5. 응답 생성
        response, needs_dashboard = self._generate_response(
            query=query,
            query_type=classification.query_type,
            context=context,
            validation_message=validation_message
        )
        
        print(f"📊 대시보드 추천: {needs_dashboard}")
        
        return ChainResult(
            query=query,
            query_type=classification.query_type,
            teams=all_teams,
            context=context,
            retrieval_score=retrieval_score,
            retrieval_method=retrieval_method,
            response=response,
            needs_dashboard=needs_dashboard,
            validation_passed=validation_passed
        )


# =============================================================================
# 싱글톤 및 편의 함수
# =============================================================================

_chain_instance: Optional[KBOAnalysisChain] = None


def get_chain() -> KBOAnalysisChain:
    """체인 싱글톤 인스턴스 반환"""
    global _chain_instance
    if _chain_instance is None:
        _chain_instance = KBOAnalysisChain()
    return _chain_instance


def run_analysis(query: str) -> ChainResult:
    """
    분석 체인을 실행하는 편의 함수
    
    Args:
        query: 사용자 쿼리
    
    Returns:
        ChainResult: 분석 결과
    
    Example:
        >>> from src.chain import run_analysis
        >>> result = run_analysis("한화 올시즌 타선 분석해줘")
        >>> print(result.response)
    """
    chain = get_chain()
    return chain.run(query)


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    # 테스트 쿼리
    test_queries = [
        "WAR가 뭐야?",
        "한화 올시즌 성적 어때?",
        "어제 LG 경기 결과 알려줘",
    ]
    
    chain = KBOAnalysisChain()
    
    for query in test_queries:
        result = chain.run(query)
        
        print(f"\n{'='*60}")
        print(f"📝 응답:")
        print(f"{'='*60}")
        print(result.response[:500] + "..." if len(result.response) > 500 else result.response)
        print(f"\n🎯 대시보드 필요: {result.needs_dashboard}")
