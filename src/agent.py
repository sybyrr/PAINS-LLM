"""
Agent module - 메인 에이전트 구현

Function Calling이 가능한 OpenAI Agent를 구현합니다.
논문 Section 4.5 (LLM Orchestration and Analysis Generation)를 구현합니다.

주요 기능:
1. 대화 관리 (메모리)
2. 도구 호출 (Function Calling)
3. 응답 생성 및 포맷팅
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .config import LLM_MODEL, OPENAI_API_KEY, TEMPERATURE
from .chain import run_analysis, ChainResult
from .tools import get_tools, generate_dashboard_json
from .utils import extract_teams_from_query, extract_date_from_query


# =============================================================================
# 시스템 프롬프트
# =============================================================================

AGENT_SYSTEM_PROMPT = """당신은 KBO 한국 프로야구 데이터 분석 전문 AI 어시스턴트입니다.

## 역할
- 사용자의 야구 관련 질문에 정확하고 통찰력 있는 답변 제공
- 팀/선수 성적 분석 및 시각화
- 야구 통계와 규칙에 대한 설명

## 핵심 원칙

1. **데이터 기반 분석**
   - 제공된 데이터만을 기반으로 분석
   - 추측이나 허구 정보 제공 금지
   - 데이터 출처 명시

2. **중립적 관점**
   - 특정 팀에 대한 편향 없이 객관적 분석
   - 긍정적/부정적 요소 균형있게 제시

3. **사용자 친화적 응답**
   - 전문 용어 사용 시 설명 추가
   - 마크다운 표와 목록 활용
   - 핵심 내용 먼저 제시

4. **도구 활용 (Function Calling)**
   - 시각화가 필요하면 generate_dashboard_json 호출
   - 팀 비교가 필요하면 compare_teams 호출
   - 도구 사용 전 사용자에게 안내

## KBO 팀 정보
- 한화 이글스 (Hanwha)
- LG 트윈스 (LG)  
- 삼성 라이온즈 (Samsung)
- 두산 베어스 (Doosan)
- 롯데 자이언츠 (Lotte)
- 기아 타이거즈 (KIA)
- NC 다이노스 (NC)
- SSG 랜더스 (SSG)
- 키움 히어로즈 (Kiwoom)
- KT 위즈 (KT)

## 응답 형식
- 분석 결과는 마크다운 표 활용
- 핵심 인사이트를 **굵은 글씨**로 강조
- 긴 응답은 섹션으로 구분

지금부터 사용자의 질문에 답변해주세요.
"""


# =============================================================================
# Agent 응답 클래스
# =============================================================================

@dataclass
class AgentResponse:
    """에이전트 응답"""
    query: str                          # 사용자 쿼리
    response: str                       # 텍스트 응답
    tool_calls: List[Dict] = field(default_factory=list)  # 호출된 도구
    dashboard: Optional[Dict] = None    # 생성된 대시보드 (있는 경우)
    context_used: Optional[Dict] = None # 사용된 컨텍스트 데이터
    error: Optional[str] = None         # 오류 메시지 (있는 경우)
    # 검색 결과 정보
    retrieval_score: float = 0.0        # 검색 유사도 점수
    retrieval_method: str = "none"      # 검색 방법 (semantic/bm25/hybrid)
    retrieved_doc_info: Optional[Dict] = None  # 검색된 문서 정보


# =============================================================================
# KBO Agent 클래스
# =============================================================================

class KBOAgent:
    """
    KBO 야구 분석 에이전트
    
    Function Calling을 지원하는 대화형 에이전트입니다.
    """
    
    def __init__(
        self, 
        model: str = None, 
        temperature: float = None,
        memory_window: int = 10
    ):
        """
        에이전트 초기화
        
        Args:
            model: 사용할 LLM 모델
            temperature: 모델 온도
            memory_window: 대화 기억 윈도우 크기
        """
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else TEMPERATURE
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=OPENAI_API_KEY
        )
        
        # 도구 로드
        self.tools = get_tools()
        
        # 메모리 초기화
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True
        )
        
        # 프롬프트 구성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 에이전트 생성
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # 에이전트 실행기
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )
    
    def chat(self, query: str, query_type: str = None) -> AgentResponse:
        """
        사용자 쿼리에 응답합니다.
        
        Args:
            query: 사용자 쿼리
            query_type: 미리 결정된 쿼리 타입 ("1", "2", "3" 또는 "general", "season_analysis", "match_analysis")
                       None이면 LLM으로 자동 분류 (기존 방식)
        
        Returns:
            AgentResponse: 에이전트 응답
        """
        try:
            # 1. 분류 수행
            if query_type:
                # 사용자가 선질문에서 선택한 타입으로 분류
                from .classifier import PreQuestionChoice, classify_by_user_choice
                
                # 사용자 입력이 "1", "2", "3" 같은 숫자인 경우 먼저 파싱
                if query_type in ["1", "2", "3"]:
                    parsed_type = PreQuestionChoice.parse_choice(query_type)
                    if parsed_type is None:
                        return AgentResponse(
                            query=query,
                            response=f"❌ 인식 불가능한 선택입니다: '{query_type}'",
                            error=f"인식 불가능한 선택입니다: '{query_type}'"
                        )
                    classification = classify_by_user_choice(query, query_type)
                else:
                    # 이미 파싱된 타입이면 직접 사용
                    if query_type not in ["general", "season_analysis", "match_analysis"]:
                        return AgentResponse(
                            query=query,
                            response=f"❌ 인식 불가능한 질문 타입입니다: '{query_type}'",
                            error=f"인식 불가능한 질문 타입입니다: '{query_type}'"
                        )
                    # 이미 파싱된 타입을 기반으로 직접 분류 결과 생성
                    from .classifier import ClassificationResult, extract_teams_from_query, extract_date_from_query
                    teams = extract_teams_from_query(query)
                    team_names = [t[0] for t in teams]
                    date = extract_date_from_query(query) if query_type in ["match_analysis", "season_analysis"] else None
                    classification = ClassificationResult(
                        reasoning_steps=f"사용자 선택: {query_type}",
                        query_type=query_type,
                        teams=team_names,
                        date=date,
                        confidence=1.0
                    )
            else:
                # 기존 방식: LLM으로 자동 분류
                from .classifier import classify_query
                classification = classify_query(query)
            
            # 2. 분석 체인 실행 (분류 결과를 전달)
            chain_result = run_analysis(query, classification)
            
            # 3. 응답 결정
            # chain에서 이미 완성된 응답을 사용 (데이터 분석 쿼리인 경우)
            if chain_result.context and chain_result.query_type != "general":
                # 분석 쿼리: chain의 응답 직접 사용 (데이터 잘림 방지)
                response_text = chain_result.response
            else:
                # 일반 질문: 에이전트 직접 처리
                result = self.agent_executor.invoke({"input": query})
                response_text = result.get("output", "")
            
            # 3. 대시보드 필요 여부 확인 및 생성
            dashboard = None
            tool_calls = []
            
            if chain_result.needs_dashboard and chain_result.teams:
                dashboard = self._create_dashboard(chain_result)
                tool_calls.append({
                    "tool": "generate_dashboard_json",
                    "args": {
                        "dashboard_type": chain_result.query_type,
                        "teams": chain_result.teams,
                    }
                })
            
            # 4. 검색된 문서 정보 구성
            retrieved_doc_info = None
            if chain_result.context:
                retrieved_doc_info = {
                    "type": chain_result.context.get("type"),
                    "teams": chain_result.context.get("teams", []),
                    "home_team": chain_result.context.get("home_team"),
                    "away_team": chain_result.context.get("away_team"),
                    "date": chain_result.context.get("date"),
                    "season": chain_result.context.get("season"),
                    "player_name": chain_result.context.get("data", {}).get("Name"),
                }
            
            return AgentResponse(
                query=query,
                response=response_text,
                tool_calls=tool_calls,
                dashboard=dashboard,
                context_used=chain_result.context,  # 실제 컨텍스트 데이터 저장
                error=None,
                retrieval_score=chain_result.retrieval_score,
                retrieval_method=chain_result.retrieval_method,
                retrieved_doc_info=retrieved_doc_info
            )
            
        except Exception as e:
            return AgentResponse(
                query=query,
                response=f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}",
                error=str(e)
            )
    
    def _enhance_query_with_context(
        self, 
        query: str, 
        chain_result: ChainResult
    ) -> str:
        """
        검색된 컨텍스트로 쿼리를 강화합니다.
        
        Args:
            query: 원본 쿼리
            chain_result: 체인 실행 결과
        
        Returns:
            str: 강화된 쿼리
        """
        context_summary = ""
        
        if chain_result.context:
            context_data = chain_result.context.get("data", {})
            context_summary = f"""

[분석 데이터]
- 데이터 유형: {chain_result.context.get('type')}
- 팀: {', '.join(chain_result.context.get('teams', []))}
- 시즌/날짜: {chain_result.context.get('season') or chain_result.context.get('date')}
- 검색 신뢰도: {chain_result.retrieval_score:.2%}

데이터 요약:
{json.dumps(context_data, ensure_ascii=False, indent=2)[:2000]}
"""
        
        return f"{query}\n{context_summary}"
    
    def _create_dashboard(self, chain_result: ChainResult) -> Optional[Dict]:
        """
        대시보드를 생성합니다.
        
        Args:
            chain_result: 체인 실행 결과
        
        Returns:
            Optional[Dict]: 대시보드 JSON
        """
        try:
            # 대시보드 유형 결정
            if chain_result.query_type == "match_analysis":
                dashboard_type = "match_analysis"
            elif len(chain_result.teams) >= 2:
                dashboard_type = "team_comparison"
            else:
                dashboard_type = "season_analysis"
            
            # 제목 생성
            teams_str = " vs ".join(chain_result.teams) if chain_result.teams else "KBO"
            title = f"{teams_str} 분석 대시보드"
            
            # season 값 추출 (None이면 기본값 사용)
            season = None
            if chain_result.context:
                season = chain_result.context.get("season")
                # 날짜에서 연도 추출 시도
                if not season and chain_result.context.get("date"):
                    date_str = chain_result.context.get("date")
                    if date_str and len(date_str) >= 4:
                        season = date_str[:4]  # "2025-06-15" -> "2025"
            season = season or "2025"  # 최종 기본값
            
            # 대시보드 생성
            dashboard = generate_dashboard_json.invoke({
                "dashboard_type": dashboard_type,
                "teams": chain_result.teams,
                "title": title,
                "date": chain_result.context.get("date") if chain_result.context else None,
                "season": season,
            })
            
            return dashboard
            
        except Exception as e:
            print(f"⚠️ 대시보드 생성 실패: {e}")
            return None
    
    def reset_memory(self):
        """대화 메모리를 초기화합니다."""
        self.memory.clear()
        print("💭 대화 기록이 초기화되었습니다.")
    
    def get_conversation_history(self) -> List[Dict]:
        """
        대화 기록을 반환합니다.
        
        Returns:
            List[Dict]: 대화 기록 목록
        """
        history = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


# =============================================================================
# 싱글톤 및 편의 함수
# =============================================================================

_agent_instance: Optional[KBOAgent] = None


def get_agent() -> KBOAgent:
    """에이전트 싱글톤 인스턴스 반환"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = KBOAgent()
    return _agent_instance


def chat(query: str, query_type: str = None) -> AgentResponse:
    """
    챗봇과 대화하는 편의 함수
    
    Args:
        query: 사용자 쿼리
        query_type: 미리 결정된 쿼리 타입 (선질문에서 받은 사용자 선택)
                   None이면 LLM으로 자동 분류
    
    Returns:
        AgentResponse: 에이전트 응답
    
    Example:
        >>> from src.agent import chat
        >>> response = chat("한화 올시즌 성적 어때?")
        >>> print(response.response)
        
        # 사용자 선택 기반 분류
        >>> response = chat("한화 성적", "2")  # 2 = 시즌 분석
        >>> print(response.response)
    """
    agent = get_agent()
    return agent.chat(query, query_type)


# =============================================================================
# 대화형 CLI
# =============================================================================

def run_interactive_chat():
    """대화형 CLI 챗봇 실행 - 사용자 질문 유형 선택 기반"""
    print("=" * 60)
    print("🎯 KBO 야구 분석 챗봇")
    print("=" * 60)
    print("명령어:")
    print("  /quit - 종료")
    print("  /reset - 대화 초기화")
    print("  /history - 대화 기록 보기")
    print("  /plot - 마지막 분석 결과 시각화")
    print("=" * 60)
    print("💡 팁: 질문 끝에 '시각화' 또는 'plot'을 추가하면 차트가 표시됩니다.")
    print("=" * 60)
    
    agent = KBOAgent()
    last_context = None
    last_query_type = None
    last_teams = None
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # 명령어 처리
            if user_input.lower() == "/quit":
                print("👋 안녕히 가세요!")
                break
            elif user_input.lower() == "/reset":
                agent.reset_memory()
                last_context = None
                last_query_type = None
                last_teams = None
                continue
            elif user_input.lower() == "/history":
                history = agent.get_conversation_history()
                print("\n📜 대화 기록:")
                for msg in history:
                    role = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{role}: {msg['content'][:100]}...")
                continue
            elif user_input.lower() == "/plot":
                # 마지막 분석 결과 시각화
                if last_context and last_query_type:
                    from .chain import get_chain
                    chain = get_chain()
                    chain._show_visualization(
                        query_type=last_query_type,
                        teams=last_teams or [],
                        context=last_context
                    )
                else:
                    print("⚠️ 시각화할 분석 결과가 없습니다. 먼저 경기나 시즌 분석 질문을 해주세요.")
                continue
            
            # "시각화", "plot", "차트" 키워드 체크
            show_plot = any(kw in user_input.lower() for kw in ['시각화', 'plot', '차트', '그래프'])
            
            # =================================================================
            # 질문 유형 선택 단계: 사용자가 제일 처음 선택
            # =================================================================
            from .classifier import generate_pre_question, PreQuestionChoice
            
            print(f"\n{generate_pre_question()}")
            
            # 사용자 선택 입력받기
            while True:
                user_choice = input("\n➡️ 선택 (1/2/3): ").strip()
                
                if not user_choice:
                    print("⚠️ 선택을 입력해주세요 (1, 2, 또는 3)")
                    continue
                
                # 유효성 검사 (1, 2, 3만 가능)
                if user_choice not in ["1", "2", "3"]:
                    print("⚠️ 인식 불가능한 선택입니다. 1, 2, 3 중 선택해주세요.")
                    continue
                
                # 유효한 선택이면 그대로 사용 (파싱은 agent.chat에서)
                query_choice = user_choice
                break  # 유효한 선택 받음
            
            # =================================================================
            # 분석 단계: 선택된 유형에 따라 처리
            # - 1 (일반 질문) → API만 호출
            # - 2 (선수 시즌 성적) → RAG 검색 + API
            # - 3 (특정 경기 분석) → RAG 검색 + API
            # =================================================================
            print("\n🤖 Assistant: ", end="", flush=True)
            response = agent.chat(user_input, query_choice)
            print(response.response)
            
            # 컨텍스트 저장
            if response.context_used and response.retrieved_doc_info:
                last_context = {
                    "type": response.retrieved_doc_info.get("type"),
                    "date": response.retrieved_doc_info.get("date"),
                    "home_team": response.retrieved_doc_info.get("home_team"),
                    "away_team": response.retrieved_doc_info.get("away_team"),
                    "teams": response.retrieved_doc_info.get("teams"),
                    "data": response.context_used.get("data", {}) if isinstance(response.context_used, dict) else {}
                }
                doc_type = response.retrieved_doc_info.get("type")
                if doc_type == "game":
                    last_query_type = "match_analysis"
                elif doc_type == "season":
                    last_query_type = "season_analysis"
                else:
                    last_query_type = doc_type
                last_teams = response.retrieved_doc_info.get("teams", [])
            
            # 검색 정보 출력 (분석 질문인 경우만)
            if response.context_used and query_choice in ["2", "3"]:
                print(f"\n📑 검색 정보 (유사도: {response.retrieval_score:.2%}, 방법: {response.retrieval_method})")
                if response.retrieved_doc_info:
                    doc = response.retrieved_doc_info
                    info_parts = []
                    if doc.get('type'):
                        info_parts.append(f"타입: {doc['type']}")
                    if doc.get('teams'):
                        info_parts.append(f"팀: {', '.join(doc['teams'])}")
                    if doc.get('date'):
                        info_parts.append(f"날짜: {doc['date']}")
                    if doc.get('player_name'):
                        info_parts.append(f"선수: {doc['player_name']}")
                    print(f"   {' | '.join(info_parts)}")
            
            # 키워드로 시각화 요청한 경우
            if show_plot and last_context and last_query_type:
                from .chain import get_chain
                chain = get_chain()
                chain._show_visualization(
                    query_type=last_query_type,
                    teams=last_teams or [],
                    context=last_context
                )
            
            # 대시보드 생성 알림
            if response.dashboard:
                print("\n📊 대시보드가 생성되었습니다!")
                print(f"   위젯 수: {len(response.dashboard.get('widgets', []))}")
                print("   💡 '/plot' 명령어로 시각화할 수 있습니다.")
                print(f"   위젯 수: {len(response.dashboard.get('widgets', []))}")
            
            # 오류 처리
            if response.error:
                print(f"\n⚠️ 오류 발생: {response.error}")
        
        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류: {e}")


# =============================================================================
# 메인 진입점
# =============================================================================

if __name__ == "__main__":
    run_interactive_chat()
