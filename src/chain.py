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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

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
    reference_doc: Optional[Dict] = None  # 참조 문서 정보
    dashboard: Optional[Dict] = None    # 대시보드 JSON (프론트엔드 렌더링용)


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

## 중요 지침

1. **모든 선수 데이터 표시 필수**: 
   - 경기 분석 시, 반드시 양 팀의 **모든 투수/타자 데이터**를 빠짐없이 표시하세요.
   - home_pitchers와 away_pitchers 모두 각각 표로 정리하세요.
   - 절대로 일부만 표시하고 생략하지 마세요.

2. **데이터 검증 우선**: 분석 전 반드시 제공된 데이터가 요청과 일치하는지 확인하세요.
   - 요청된 팀과 데이터의 팀이 일치하는가?
   - 요청된 기간(시즌/경기 날짜)이 데이터와 일치하는가?

3. **불일치 처리**:
   - 완전 일치: 분석 진행
   - 부분 일치 (팀은 맞지만 기간 불일치): 사용자에게 알리고 가용 데이터로 분석
   - 팀 불일치: 분석 거부, 올바른 데이터 요청 안내

4. **응답 형식**:
   - 마크다운 표를 활용한 명확한 데이터 표시
   - 핵심 인사이트를 먼저, 상세 분석은 뒤에
   - 수치는 반드시 데이터 기반으로
   - 경기 분석시, 상세 분석을 표시할 때 모든 투수 데이터를 활용하도록 하세요.

5. **시각화 필요성 판단**:
   - 데이터가 풍부하고 비교 분석이 필요하면 대시보드 추천
   - 단순 질문이나 데이터 부족시 텍스트 답변으로 충분

6. 만약 시즌 분석시 선수에 대한 특정 지표만 요청한다면 반드시 해당 지표와 그에 대한 분석만을 출력하세요.

7. **시즌 분석의 심화 지표 분석 (선수 자체의 분석을 요청한다면 시즌 분석의 핵심 인사이트에 포함)**:
   투수 분석 시 특정 지표 언급 없이 선수에 대한 분석을 요청한다면 기본 지표(ERA, W-L, IP, SO) 외에 다음 심화 지표를 반드시 분석하세요:
   
   **효율성**:
   - PIP (이닝당 투구수): 선발 투수의 경우에만 분석. 낮을수록 효율적. (16 이하면 효율적, 17.9 이상이면 비효율)
   - LOB (잔루율): 73%보다 높으면 리그 내에서 위기 탈출 능력이 우수한 투수로 판단 가능. 73%는 리그 평균이 아닌 상위 25% 수준임.
   
   **피안타/피홈런 분석**:
   - OPS (피OPS): 상대 타자에게 허용한 OPS, 낮을수록 좋음 (0.620 이하 우수, 0.800 이상이면 주의)
   - BABIP (인플레이 피안타율): 운/수비 영향, 리그 중앙값 약 .300
   
   **제구력/탈삼진 능력**:
   - K9 (9이닝당 삼진률): 9.5 이상이라면 삼진 능력이 우수한 투수로 판단 가능
   - BB9 (9이닝당 볼넷률): 2.3 이하라면 제구가 좋은 투수로 판단 가능  
   - WHIP (이닝당 출루 허용): 낮을수록 좋음 (1.25 이하 우수, 1.40 이상 주의)
   
   **실력 평가 지표**:
   - FIP (수비 무관 평균자책점): 투수 본연의 실력, ERA와 비교 분석. FIP가 ERA보다 유의미하게 낮으면 투수가 수비 도움을 잘 받지 못했다고 판단 가능.
   - xFIP (기대 FIP): 뜬공이 홈런 되는 확률도 공평하게 평균으로 맞추어 산정. xFIP가 FIP보다 높으면 투수가 홈런 운이 좋았다고 판단 가능.
   - WAR (대체선수 대비 승리 기여도): 종합 가치 평가 (음수면 대체선수보다 못함)
   
   핵심 인사이트 작성 시 위 지표들을 종합하여 선수의 강점/약점을 분석하세요.
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
            api_key=OPENAI_API_KEY,
            max_tokens=4096  # 충분한 출력 길이 보장
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
            # 경기 데이터의 경우 투수별로 명확하게 정리
            formatted_data = self._format_context_for_prompt(context)
            
            user_content = f"""## 사용자 질문
{query}

## 데이터 검증 상태
{validation_message}

## 분석 데이터
{formatted_data}

위 데이터를 기반으로 분석해주세요.
- 반드시 제공된 모든 선수 데이터를 표로 정리해서 보여주세요.
- 데이터가 요청과 일치하지 않으면 그 사실을 먼저 알려주세요.
- 대시보드(시각화)가 유용할 것 같으면 마지막에 "[대시보드 추천]"을 포함해주세요.
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
    
    def _format_context_for_prompt(self, context: Dict) -> str:
        """
        LLM 프롬프트용으로 컨텍스트를 명확한 형식으로 포맷팅합니다.
        
        경기 데이터의 경우 양 팀 투수를 각각 정리합니다.
        """
        data = context.get("data", {})
        data_type = context.get("type")
        
        if data_type == "game":
            # 경기 데이터: 양 팀 투수 명확히 분리
            home_team = context.get("home_team", "홈팀")
            away_team = context.get("away_team", "원정팀")
            date = context.get("date", "")
            
            lines = []
            lines.append(f"### 경기 정보")
            lines.append(f"- 날짜: {date}")
            lines.append(f"- 홈팀: {home_team} (점수: {data.get('home_runs', 'N/A')})")
            lines.append(f"- 원정팀: {away_team} (점수: {data.get('away_runs', 'N/A')})")
            lines.append(f"- 시즌 유형: {data.get('season_type', 'N/A')}")
            lines.append("")
            
            # 홈팀 투수 목록
            home_pitchers = data.get("home_pitchers", [])
            lines.append(f"### {home_team} 투수진 ({len(home_pitchers)}명)")
            if home_pitchers:
                lines.append("| 이름 | 이닝 | 자책점 | 삼진 | 피안타 | 볼넷 | 결과 |")
                lines.append("|------|------|--------|------|--------|------|------|")
                for p in home_pitchers:
                    name = p.get("Name", "N/A")
                    ip = p.get("IP", "N/A")
                    er = p.get("ER", "N/A")
                    so = p.get("SO", "N/A")
                    h = p.get("H", "N/A")
                    bb = p.get("BB", "N/A")
                    result = p.get("Result", "-")
                    lines.append(f"| {name} | {ip} | {er} | {so} | {h} | {bb} | {result or '-'} |")
            lines.append("")
            
            # 원정팀 투수 목록
            away_pitchers = data.get("away_pitchers", [])
            lines.append(f"### {away_team} 투수진 ({len(away_pitchers)}명)")
            if away_pitchers:
                lines.append("| 이름 | 이닝 | 자책점 | 삼진 | 피안타 | 볼넷 | 결과 |")
                lines.append("|------|------|--------|------|--------|------|------|")
                for p in away_pitchers:
                    name = p.get("Name", "N/A")
                    ip = p.get("IP", "N/A")
                    er = p.get("ER", "N/A")
                    so = p.get("SO", "N/A")
                    h = p.get("H", "N/A")
                    bb = p.get("BB", "N/A")
                    result = p.get("Result", "-")
                    lines.append(f"| {name} | {ip} | {er} | {so} | {h} | {bb} | {result or '-'} |")
            
            return "\n".join(lines)
        
        elif data_type == "season":
            # 시즌 데이터: 선수 개인 시즌 통계를 구조화하여 전달
            lines = []
            lines.append("### 선수 시즌 통계")
            lines.append("")
            
            # 기본 정보
            name = data.get("Name", "N/A")
            team = data.get("Team", "N/A")
            season = data.get("Season", data.get("season", "N/A"))
            season_type = data.get("Season_Type", data.get("_season_type", "Regular"))
            
            lines.append(f"**선수명**: {name}")
            lines.append(f"**소속팀**: {team}")
            lines.append(f"**시즌**: {season} {season_type}")
            lines.append("")
            
            # 기본 성적
            lines.append("#### 기본 성적")
            lines.append("| 지표 | 값 | 설명 |")
            lines.append("|------|-----|------|")
            lines.append(f"| ERA | {data.get('ERA', 'N/A')} | 평균자책점 |")
            lines.append(f"| W-L | {data.get('W', 0)}-{data.get('L', 0)} | 승-패 |")
            lines.append(f"| G/GS | {data.get('G', 0)}/{data.get('GS', 0)} | 경기수/선발경기수 |")
            lines.append(f"| IP | {data.get('IP', 'N/A')} | 이닝 |")
            lines.append(f"| SO | {data.get('SO', 'N/A')} | 삼진 |")
            lines.append(f"| BB | {data.get('BB', 'N/A')} | 볼넷 |")
            lines.append(f"| H | {data.get('H', 'N/A')} | 피안타 |")
            lines.append(f"| HR | {data.get('HR', 'N/A')} | 피홈런 |")
            lines.append(f"| S | {data.get('S', 'N/A')} | 세이브 |")
            lines.append(f"| HD | {data.get('HD', 'N/A')} | 홀드 |")
            lines.append("")
            
            # 효율성 지표
            lines.append("#### 효율성 지표")
            lines.append("| 지표 | 값 | 설명 |")
            lines.append("|------|-----|------|")
            lines.append(f"| PPA | {data.get('PPA', 'N/A')} | 타자당 투구수 (낮을수록 효율적) |")
            lines.append(f"| PIP | {data.get('PIP', 'N/A')} | 이닝당 투구수 (낮을수록 효율적) |")
            lines.append(f"| NP | {data.get('NP', 'N/A')} | 총 투구수 |")
            lines.append(f"| PG | {data.get('PG', 'N/A')} | 경기당 투구수 |")
            lines.append("")
            
            # 피안타/피홈런 지표
            lines.append("#### 피안타/피홈런 분석")
            lines.append("| 지표 | 값 | 설명 |")
            lines.append("|------|-----|------|")
            lines.append(f"| 피OPS | {data.get('OPS', 'N/A')} | 상대 타자 출루율+장타율 (낮을수록 좋음) |")
            lines.append(f"| 피타율 | {data.get('AVG', 'N/A')} | 상대 타자에게 허용한 타율 |")
            lines.append(f"| 피출루율 | {data.get('OBP', 'N/A')} | 상대 타자에게 허용한 출루율 |")
            lines.append(f"| 피장타율 | {data.get('SLG', 'N/A')} | 상대 타자에게 허용한 장타율 |")
            lines.append(f"| BABIP | {data.get('BABIP', 'N/A')} | 인플레이 피안타율 (운/수비 영향) |")
            lines.append(f"| HR9 | {data.get('HR9', 'N/A')} | 9이닝당 피홈런 |")
            lines.append("")
            
            # 제구력/탈삼진 지표
            lines.append("#### 제구력/탈삼진 분석")
            lines.append("| 지표 | 값 | 설명 |")
            lines.append("|------|-----|------|")
            lines.append(f"| K% | {data.get('K%', 'N/A')} | 삼진률 (높을수록 좋음) |")
            lines.append(f"| BB% | {data.get('BB%', 'N/A')} | 볼넷률 (낮을수록 좋음) |")
            lines.append(f"| K-BB% | {data.get('KminusBB%', 'N/A')} | 삼진-볼넷 비율 (높을수록 좋음) |")
            lines.append(f"| K9 | {data.get('K9', 'N/A')} | 9이닝당 삼진 |")
            lines.append(f"| BB9 | {data.get('BB9', 'N/A')} | 9이닝당 볼넷 |")
            lines.append(f"| WHIP | {data.get('WHIP', 'N/A')} | 이닝당 출루 허용 (낮을수록 좋음) |")
            lines.append("")
            
            # 실력 평가 지표
            lines.append("#### 실력 평가 지표")
            lines.append("| 지표 | 값 | 설명 |")
            lines.append("|------|-----|------|")
            lines.append(f"| FIP | {data.get('FIP', 'N/A')} | 수비 무관 평균자책점 (투수 본연의 실력) |")
            lines.append(f"| xFIP | {data.get('xFIP', 'N/A')} | 기대 FIP (홈런 운 배제) |")
            lines.append(f"| ERA-FIP | {data.get('ERAminusFIP', 'N/A')} | ERA와 FIP 차이 (양수=불운, 음수=행운) |")
            lines.append(f"| WAR | {data.get('WAR', 'N/A')} | 대체선수 대비 승리 기여도 |")
            lines.append(f"| LOB% | {data.get('LOB', 'N/A')} | 잔루율 (위기 탈출 능력) |")
            lines.append("")
            
            return "\n".join(lines)
        
        else:
            # 기타 데이터: JSON 그대로 전달
            return f"```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```"
    
    def _create_dashboard(
        self,
        query_type: str,
        teams: List[str],
        context: Dict
    ) -> Optional[Dict]:
        """
        프론트엔드 렌더링용 대시보드 JSON을 생성합니다.
        
        논문 Section 4.5.4 (Function Calling for Dashboard Generation):
        LLM은 차트를 직접 그리지 않고, 프론트엔드가 렌더링할 JSON 설계도를 생성합니다.
        
        Args:
            query_type: 쿼리 유형
            teams: 팀 목록
            context: 검색된 컨텍스트
        
        Returns:
            Dict: 대시보드 JSON (Streamlit/React에서 렌더링)
        """
        try:
            data = context.get("data", {})
            date = context.get("date", "")
            
            if query_type == "match_analysis":
                # 경기 분석 대시보드
                home_team = context.get("home_team", "")
                away_team = context.get("away_team", "")
                
                home_pitchers = data.get("home_pitchers", [])
                away_pitchers = data.get("away_pitchers", [])
                
                dashboard = {
                    "type": "match_analysis",
                    "title": f"{home_team} vs {away_team} 경기 분석",
                    "date": date,
                    "metadata": {
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_runs": data.get("home_runs"),
                        "away_runs": data.get("away_runs"),
                        "season_type": data.get("season_type")
                    },
                    "widgets": [
                        {
                            "id": "score_summary",
                            "type": "score_card",
                            "title": "경기 스코어",
                            "data": {
                                "home": {"team": home_team, "runs": data.get("home_runs")},
                                "away": {"team": away_team, "runs": data.get("away_runs")}
                            }
                        },
                        {
                            "id": "home_pitchers",
                            "type": "table",
                            "title": f"{home_team} 투수진",
                            "columns": ["이름", "이닝", "자책점", "삼진", "피안타", "볼넷", "결과"],
                            "data": [
                                {
                                    "이름": p.get("Name"),
                                    "이닝": p.get("IP"),
                                    "자책점": p.get("ER"),
                                    "삼진": p.get("SO"),
                                    "피안타": p.get("H"),
                                    "볼넷": p.get("BB"),
                                    "결과": p.get("Result") or "-"
                                }
                                for p in home_pitchers
                            ]
                        },
                        {
                            "id": "away_pitchers",
                            "type": "table",
                            "title": f"{away_team} 투수진",
                            "columns": ["이름", "이닝", "자책점", "삼진", "피안타", "볼넷", "결과"],
                            "data": [
                                {
                                    "이름": p.get("Name"),
                                    "이닝": p.get("IP"),
                                    "자책점": p.get("ER"),
                                    "삼진": p.get("SO"),
                                    "피안타": p.get("H"),
                                    "볼넷": p.get("BB"),
                                    "결과": p.get("Result") or "-"
                                }
                                for p in away_pitchers
                            ]
                        },
                        {
                            "id": "pitching_comparison",
                            "type": "bar_chart",
                            "title": "투수진 비교",
                            "x_axis": "팀",
                            "y_axis": "자책점 합계",
                            "data": [
                                {"팀": home_team, "자책점": sum(p.get("ER", 0) or 0 for p in home_pitchers)},
                                {"팀": away_team, "자책점": sum(p.get("ER", 0) or 0 for p in away_pitchers)}
                            ]
                        }
                    ]
                }
                
            elif query_type == "season_analysis":
                # 시즌 분석 대시보드
                player_name = data.get("Name", "")
                team = context.get("teams", [""])[0] if context.get("teams") else ""
                
                dashboard = {
                    "type": "season_analysis",
                    "title": f"{player_name} 시즌 성적",
                    "metadata": {
                        "player": player_name,
                        "team": team,
                        "season": data.get("season", "2025")
                    },
                    "widgets": [
                        {
                            "id": "player_stats",
                            "type": "stat_card",
                            "title": "주요 지표",
                            "data": {
                                "ERA": data.get("ERA"),
                                "WHIP": data.get("WHIP"),
                                "승": data.get("W"),
                                "패": data.get("L"),
                                "이닝": data.get("IP"),
                                "삼진": data.get("SO")
                            }
                        }
                    ]
                }
            else:
                return None
            
            print(f"📊 대시보드 생성 완료: {len(dashboard.get('widgets', []))}개 위젯")
            return dashboard
            
        except Exception as e:
            print(f"⚠️ 대시보드 생성 실패: {e}")
            return None
    
    def _show_visualization(
        self,
        query_type: str,
        teams: List[str],
        context: Dict
    ) -> None:
        """
        matplotlib을 사용해서 별도 창으로 시각화를 표시합니다.
        
        Args:
            query_type: 쿼리 유형
            teams: 팀 목록
            context: 검색된 컨텍스트
        """
        try:
            # 한글 폰트 설정 (Windows)
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            
            data = context.get("data", {})
            date = context.get("date", "")
            
            if query_type in ("match_analysis", "game"):
                self._plot_match_analysis(context, data, date)
            elif query_type in ("season_analysis", "season"):
                self._plot_season_analysis(context, data)
            else:
                print(f"⚠️ 시각화 미지원 쿼리 유형입니다: {query_type}")
                return
            
            plt.tight_layout()
            plt.show()
            print("📊 시각화 창이 표시되었습니다.")
            
        except Exception as e:
            print(f"⚠️ 시각화 실패: {e}")
    
    def _plot_match_analysis(self, context: Dict, data: Dict, date: str) -> None:
        """경기 분석 시각화"""
        home_team = context.get("home_team", "홈팀")
        away_team = context.get("away_team", "원정팀")
        home_pitchers = data.get("home_pitchers", [])
        away_pitchers = data.get("away_pitchers", [])
        
        # 숫자 변환 헬퍼
        def to_int(val):
            try:
                return int(val) if val is not None else 0
            except (ValueError, TypeError):
                return 0
        
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{date} {home_team} vs {away_team} 경기 분석", fontsize=16, fontweight='bold')
        
        # 1. 스코어 카드 (좌상단)
        ax1 = axes[0, 0]
        home_runs = to_int(data.get("home_runs", 0))
        away_runs = to_int(data.get("away_runs", 0))
        teams = [home_team, away_team]
        runs = [home_runs, away_runs]
        colors = ['#1f77b4', '#ff7f0e']
        
        bars = ax1.bar(teams, runs, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('득점', fontsize=12)
        ax1.set_title('경기 스코어', fontsize=14)
        for bar, run in zip(bars, runs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(int(run)), ha='center', va='bottom', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(runs) * 1.3 if max(runs) > 0 else 5)
        
        # 2. 투수별 자책점 비교 (우상단)
        ax2 = axes[0, 1]
        all_pitchers = []
        all_er = []
        all_colors = []
        
        for p in home_pitchers:
            all_pitchers.append(p.get("Name", "?"))
            all_er.append(to_int(p.get("ER", 0)))
            all_colors.append('#1f77b4')
        for p in away_pitchers:
            all_pitchers.append(p.get("Name", "?"))
            all_er.append(to_int(p.get("ER", 0)))
            all_colors.append('#ff7f0e')
        
        if all_pitchers:
            y_pos = np.arange(len(all_pitchers))
            ax2.barh(y_pos, all_er, color=all_colors, edgecolor='black')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(all_pitchers)
            ax2.set_xlabel('자책점 (ER)', fontsize=12)
            ax2.set_title('투수별 자책점', fontsize=14)
            ax2.legend([plt.Rectangle((0,0),1,1,fc='#1f77b4'), 
                       plt.Rectangle((0,0),1,1,fc='#ff7f0e')], 
                      [home_team, away_team], loc='lower right')
        
        # 3. 투수별 삼진 수 (좌하단)
        ax3 = axes[1, 0]
        all_so = []
        for p in home_pitchers:
            all_so.append(to_int(p.get("SO", 0)))
        for p in away_pitchers:
            all_so.append(to_int(p.get("SO", 0)))
        
        if all_pitchers:
            y_pos = np.arange(len(all_pitchers))
            ax3.barh(y_pos, all_so, color=all_colors, edgecolor='black')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(all_pitchers)
            ax3.set_xlabel('삼진 (SO)', fontsize=12)
            ax3.set_title('투수별 삼진', fontsize=14)
        
        # 4. 팀별 투구 이닝 합계 (우하단)
        ax4 = axes[1, 1]
        
        def parse_ip(ip_str):
            """이닝 문자열을 숫자로 변환 (예: '5.2' -> 5.67)"""
            try:
                if ip_str is None:
                    return 0
                ip_str = str(ip_str)
                if '.' in ip_str:
                    whole, frac = ip_str.split('.')
                    return int(whole) + int(frac) / 3
                return float(ip_str)
            except:
                return 0
        
        home_ip = sum(parse_ip(p.get("IP")) for p in home_pitchers)
        away_ip = sum(parse_ip(p.get("IP")) for p in away_pitchers)
        home_er_total = sum(to_int(p.get("ER")) for p in home_pitchers)
        away_er_total = sum(to_int(p.get("ER")) for p in away_pitchers)
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, [home_ip, away_ip], width, label='투구 이닝', color='#2ecc71')
        bars2 = ax4.bar(x + width/2, [home_er_total, away_er_total], width, label='자책점 합계', color='#e74c3c')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([home_team, away_team])
        ax4.set_ylabel('값', fontsize=12)
        ax4.set_title('팀별 투수 성적 요약', fontsize=14)
        ax4.legend()
        
        # 값 표시
        for bar in bars1:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)
    
    def _plot_season_analysis(self, context: Dict, data: Dict) -> None:
        """시즌 분석 시각화"""
        player_name = data.get("Name", "선수")
        team = context.get("teams", [""])[0] if context.get("teams") else ""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{player_name} ({team}) 시즌 성적", fontsize=16, fontweight='bold')
        
        # 1. 주요 지표 바 차트
        ax1 = axes[0]
        metrics = ['ERA', 'WHIP', 'W', 'L', 'SO']
        values = [data.get(m, 0) or 0 for m in metrics]
        
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax1.bar(metrics, values, color=colors, edgecolor='black')
        ax1.set_ylabel('값', fontsize=12)
        ax1.set_title('주요 성적 지표', fontsize=14)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}' if isinstance(val, float) else str(val),
                    ha='center', va='bottom', fontsize=10)
        
        # 2. 투구 이닝 및 삼진 관계
        ax2 = axes[1]
        ip = data.get("IP", 0) or 0
        so = data.get("SO", 0) or 0
        bb = data.get("BB", 0) or 0
        
        categories = ['이닝 (IP)', '삼진 (SO)', '볼넷 (BB)']
        vals = [ip, so, bb]
        colors2 = ['#1abc9c', '#e74c3c', '#3498db']
        
        bars2 = ax2.bar(categories, vals, color=colors2, edgecolor='black')
        ax2.set_ylabel('값', fontsize=12)
        ax2.set_title('투구 세부 지표', fontsize=14)
        
        for bar, val in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom', fontsize=10)
    
    def run(self, query: str, classification: Optional[ClassificationResult] = None, show_plot: bool = False) -> ChainResult:
        """
        전체 체인을 실행합니다.
        
        Args:
            query: 사용자 쿼리
            classification: 미리 분류된 결과 (선질문에서 받은 사용자 선택)
                           None이면 LLM으로 자동 분류 (기존 방식)
            show_plot: True이면 matplotlib 창으로 시각화 표시
        
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
        if classification is None:
            # 자동 분류 (API 호출)
            classification = classify_query(query)
            print(f"🏷️ 분류 (자동): {classification.query_type} (신뢰도: {classification.confidence:.2f})")
        else:
            # 사용자 선택 기반 분류 (API 호출 없음)
            print(f"🏷️ 분류 (사용자 선택): {classification.query_type} (신뢰도: {classification.confidence:.2f})")
        
        print(f"📅 날짜: {classification.date}")
        
        # 팀 정보 병합 (분류기 + 정규화)
        all_teams = list(set(normalized_teams + classification.teams))
        
        # 3. 검색 (분석 쿼리인 경우만)
        context = None
        retrieval_score = 0.0
        retrieval_method = "none"
        validation_passed = True
        validation_message = ""
        reference_doc = None
        
        if classification.query_type != "general":
            context, retrieval_score, retrieval_method = retrieve_for_query(
                query=query,
                query_type=classification.query_type,
                teams=all_teams,
                date=classification.date
            )
            
            print(f"🔍 검색 결과: {retrieval_method} (점수: {retrieval_score:.4f})")
            
            # 참조 문서 정보 저장
            if context:
                reference_doc = {
                    "type": context.get("type"),
                    "date": context.get("date"),
                    "home_team": context.get("home_team"),
                    "away_team": context.get("away_team"),
                    "teams": context.get("teams"),
                    "score": retrieval_score,
                    "method": retrieval_method
                }
            
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
        
        # 6. 대시보드 생성 (필요한 경우)
        dashboard = None
        if needs_dashboard and all_teams and context:
            dashboard = self._create_dashboard(
                query_type=classification.query_type,
                teams=all_teams,
                context=context
            )
        
        # 7. matplotlib 시각화 표시 (show_plot=True인 경우)
        if show_plot and context and classification.query_type != "general":
            self._show_visualization(
                query_type=classification.query_type,
                teams=all_teams,
                context=context
            )
        
        return ChainResult(
            query=query,
            query_type=classification.query_type,
            teams=all_teams,
            context=context,
            retrieval_score=retrieval_score,
            retrieval_method=retrieval_method,
            response=response,
            needs_dashboard=needs_dashboard,
            validation_passed=validation_passed,
            reference_doc=reference_doc,
            dashboard=dashboard
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


def run_analysis(query: str, classification: Optional[ClassificationResult] = None, show_plot: bool = False) -> ChainResult:
    """
    분석 체인을 실행하는 편의 함수
    
    Args:
        query: 사용자 쿼리
        classification: 미리 분류된 결과 (선질문에서 받은 사용자 선택)
                       None이면 LLM으로 자동 분류 (기존 방식)
        show_plot: True이면 matplotlib 창으로 시각화 표시
    
    Returns:
        ChainResult: 분석 결과
    
    Example:
        >>> from src.chain import run_analysis
        >>> result = run_analysis("한화 올시즌 타선 분석해줘")
        >>> print(result.response)
        
        # 사용자 선택 기반 분류
        >>> from src.classifier import classify_by_user_choice
        >>> user_classification = classify_by_user_choice("한화 성적", "2")
        >>> result = run_analysis("한화 성적", user_classification)
        
        # matplotlib 창으로 시각화
        >>> result = run_analysis("6월 25일 롯데 NC 경기 분석해줘", show_plot=True)
    """
    chain = get_chain()
    
    # 분류 결과가 없으면 자동 분류
    if classification is None:
        classification = classify_query(query)
    
    return chain.run(query, classification, show_plot=show_plot)


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
