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
    context_used: bool = False          # 검색된 컨텍스트 사용 여부
    error: Optional[str] = None         # 오류 메시지 (있는 경우)


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
    
    def chat(self, query: str) -> AgentResponse:
        """
        사용자 쿼리에 응답합니다.
        
        Args:
            query: 사용자 쿼리
        
        Returns:
            AgentResponse: 에이전트 응답
        """
        try:
            # 1. 분석 체인 실행 (분류 + 검색)
            chain_result = run_analysis(query)
            
            # 2. 컨텍스트 기반 에이전트 호출
            if chain_result.context:
                # 검색된 데이터가 있는 경우 컨텍스트 포함
                enhanced_query = self._enhance_query_with_context(
                    query, chain_result
                )
                result = self.agent_executor.invoke({"input": enhanced_query})
            else:
                # 일반 질문인 경우 직접 처리
                result = self.agent_executor.invoke({"input": query})
            
            # 3. 응답 파싱
            response_text = result.get("output", "")
            
            # 4. 대시보드 필요 여부 확인 및 생성
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
            
            return AgentResponse(
                query=query,
                response=response_text,
                tool_calls=tool_calls,
                dashboard=dashboard,
                context_used=chain_result.context is not None,
                error=None
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
            
            # 대시보드 생성
            dashboard = generate_dashboard_json.invoke({
                "dashboard_type": dashboard_type,
                "teams": chain_result.teams,
                "title": title,
                "date": chain_result.context.get("date") if chain_result.context else None,
                "season": chain_result.context.get("season", "2025") if chain_result.context else "2025",
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


def chat(query: str) -> AgentResponse:
    """
    챗봇과 대화하는 편의 함수
    
    Args:
        query: 사용자 쿼리
    
    Returns:
        AgentResponse: 에이전트 응답
    
    Example:
        >>> from src.agent import chat
        >>> response = chat("한화 올시즌 성적 어때?")
        >>> print(response.response)
    """
    agent = get_agent()
    return agent.chat(query)


# =============================================================================
# 대화형 CLI
# =============================================================================

def run_interactive_chat():
    """대화형 CLI 챗봇 실행"""
    print("=" * 60)
    print("🎯 KBO 야구 분석 챗봇")
    print("=" * 60)
    print("명령어:")
    print("  /quit - 종료")
    print("  /reset - 대화 초기화")
    print("  /history - 대화 기록 보기")
    print("=" * 60)
    
    agent = KBOAgent()
    
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
                continue
            elif user_input.lower() == "/history":
                history = agent.get_conversation_history()
                print("\n📜 대화 기록:")
                for msg in history:
                    role = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{role}: {msg['content'][:100]}...")
                continue
            
            # 챗봇 응답
            print("\n🤖 Assistant: ", end="")
            response = agent.chat(user_input)
            print(response.response)
            
            # 대시보드 생성 알림
            if response.dashboard:
                print("\n📊 대시보드가 생성되었습니다!")
                print(f"   위젯 수: {len(response.dashboard.get('widgets', []))}")
            
            # 도구 호출 정보
            if response.tool_calls:
                print(f"\n🔧 사용된 도구: {[t['tool'] for t in response.tool_calls]}")
                
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
