"""
Tools module - Function Calling 도구 정의

LLM이 호출할 수 있는 구조화된 도구(Function)를 정의합니다.
논문 Section 4.5.4 (Function Calling for Dashboard Generation)를 구현합니다.

주요 도구:
1. generate_dashboard_json - 프론트엔드용 대시보드 JSON 생성
2. get_team_statistics - 팀 통계 조회
3. compare_teams - 팀 비교 분석
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.tools import tool, StructuredTool
from langchain_core.tools import BaseTool


# =============================================================================
# 입력 스키마 정의
# =============================================================================

class DashboardInput(BaseModel):
    """대시보드 생성 입력 스키마"""
    dashboard_type: str = Field(
        description="대시보드 유형: 'season_analysis', 'match_analysis', 'team_comparison'"
    )
    teams: List[str] = Field(
        description="분석 대상 팀 목록 (정규화된 영문명)"
    )
    title: str = Field(
        description="대시보드 제목"
    )
    date: Optional[str] = Field(
        default=None,
        description="경기 날짜 (match_analysis인 경우, YYYY-MM-DD)"
    )
    season: str = Field(
        default="2025",
        description="시즌 연도"
    )
    metrics: List[str] = Field(
        default=["타율", "OPS", "ERA", "WAR"],
        description="표시할 지표 목록"
    )


class TeamStatsInput(BaseModel):
    """팀 통계 조회 입력 스키마"""
    team: str = Field(description="팀명 (정규화된 영문명)")
    season: str = Field(default="2025", description="시즌 연도")
    stat_category: str = Field(
        default="batting",
        description="통계 카테고리: 'batting', 'pitching', 'fielding'"
    )


class TeamComparisonInput(BaseModel):
    """팀 비교 입력 스키마"""
    team1: str = Field(description="첫 번째 팀명")
    team2: str = Field(description="두 번째 팀명")
    metrics: List[str] = Field(
        default=["승률", "타율", "평균자책점"],
        description="비교할 지표 목록"
    )


# =============================================================================
# 대시보드 생성 도구
# =============================================================================

@tool
def generate_dashboard_json(
    dashboard_type: str,
    teams: List[str],
    title: str,
    date: Optional[str] = None,
    season: str = "2025",
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    프론트엔드용 대시보드 JSON을 생성합니다.
    
    시각화가 필요한 분석 결과를 구조화된 JSON으로 반환합니다.
    프론트엔드에서 이 JSON을 받아 차트/테이블을 렌더링합니다.
    
    Args:
        dashboard_type: 대시보드 유형 ('season_analysis', 'match_analysis', 'team_comparison')
        teams: 분석 대상 팀 목록
        title: 대시보드 제목
        date: 경기 날짜 (match_analysis인 경우)
        season: 시즌 연도
        metrics: 표시할 지표 목록
    
    Returns:
        대시보드 구조를 담은 JSON 딕셔너리
    """
    if metrics is None:
        metrics = ["타율", "OPS", "ERA", "WAR"]
    
    # 기본 대시보드 구조
    dashboard = {
        "id": f"dashboard_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "type": dashboard_type,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "metadata": {
            "teams": teams,
            "season": season,
            "date": date,
        },
        "widgets": []
    }
    
    # 대시보드 유형별 위젯 구성
    if dashboard_type == "season_analysis":
        dashboard["widgets"] = _create_season_widgets(teams, season, metrics)
    elif dashboard_type == "match_analysis":
        dashboard["widgets"] = _create_match_widgets(teams, date, metrics)
    elif dashboard_type == "team_comparison":
        dashboard["widgets"] = _create_comparison_widgets(teams, metrics)
    
    return dashboard


def _create_season_widgets(
    teams: List[str], 
    season: str, 
    metrics: List[str]
) -> List[Dict]:
    """시즌 분석용 위젯 생성"""
    widgets = []
    
    # 1. 팀 개요 카드
    for team in teams:
        widgets.append({
            "id": f"overview_{team}",
            "type": "stat_card",
            "title": f"{team} {season} 시즌 개요",
            "config": {
                "team": team,
                "season": season,
                "metrics": ["승", "패", "무", "승률", "순위"],
            }
        })
    
    # 2. 성적 추이 차트
    widgets.append({
        "id": "performance_trend",
        "type": "line_chart",
        "title": "월별 성적 추이",
        "config": {
            "teams": teams,
            "x_axis": "month",
            "y_axis": "win_rate",
            "season": season,
        }
    })
    
    # 3. 주요 지표 바 차트
    widgets.append({
        "id": "key_metrics",
        "type": "bar_chart",
        "title": "주요 지표 비교",
        "config": {
            "teams": teams,
            "metrics": metrics,
        }
    })
    
    # 4. 선수 랭킹 테이블
    widgets.append({
        "id": "player_rankings",
        "type": "data_table",
        "title": "주요 선수 성적",
        "config": {
            "teams": teams,
            "columns": ["선수명", "포지션", "타율", "홈런", "타점", "WAR"],
            "sort_by": "WAR",
            "limit": 10,
        }
    })
    
    return widgets


def _create_match_widgets(
    teams: List[str], 
    date: Optional[str], 
    metrics: List[str]
) -> List[Dict]:
    """경기 분석용 위젯 생성"""
    widgets = []
    
    home_team = teams[0] if teams else "Home"
    away_team = teams[1] if len(teams) > 1 else "Away"
    
    # 1. 경기 스코어보드
    widgets.append({
        "id": "scoreboard",
        "type": "scoreboard",
        "title": f"{home_team} vs {away_team}",
        "config": {
            "home_team": home_team,
            "away_team": away_team,
            "date": date,
            "show_innings": True,
        }
    })
    
    # 2. 팀별 주요 스탯 비교
    widgets.append({
        "id": "team_comparison",
        "type": "comparison_chart",
        "title": "팀 스탯 비교",
        "config": {
            "teams": [home_team, away_team],
            "metrics": ["안타", "홈런", "삼진", "볼넷", "실책"],
        }
    })
    
    # 3. 이닝별 득점
    widgets.append({
        "id": "innings_score",
        "type": "stacked_bar",
        "title": "이닝별 득점",
        "config": {
            "teams": [home_team, away_team],
            "x_axis": "inning",
            "y_axis": "runs",
        }
    })
    
    # 4. 주요 플레이 타임라인
    widgets.append({
        "id": "key_plays",
        "type": "timeline",
        "title": "주요 플레이",
        "config": {
            "event_types": ["홈런", "도루", "더블플레이", "득점"],
        }
    })
    
    # 5. 투수 성적
    widgets.append({
        "id": "pitching_stats",
        "type": "data_table",
        "title": "투수 성적",
        "config": {
            "columns": ["투수명", "팀", "이닝", "피안타", "삼진", "볼넷", "실점"],
        }
    })
    
    return widgets


def _create_comparison_widgets(
    teams: List[str], 
    metrics: List[str]
) -> List[Dict]:
    """팀 비교용 위젯 생성"""
    widgets = []
    
    # 1. 레이더 차트
    widgets.append({
        "id": "radar_comparison",
        "type": "radar_chart",
        "title": "팀 역량 비교",
        "config": {
            "teams": teams,
            "metrics": ["공격력", "수비력", "투수력", "주루", "클러치"],
        }
    })
    
    # 2. 지표별 막대 차트
    for metric in metrics:
        widgets.append({
            "id": f"metric_{metric}",
            "type": "horizontal_bar",
            "title": f"{metric} 비교",
            "config": {
                "teams": teams,
                "metric": metric,
            }
        })
    
    # 3. 상대 전적
    if len(teams) >= 2:
        widgets.append({
            "id": "head_to_head",
            "type": "stat_card",
            "title": "시즌 상대 전적",
            "config": {
                "team1": teams[0],
                "team2": teams[1],
                "show_history": True,
            }
        })
    
    return widgets


# =============================================================================
# 팀 통계 조회 도구
# =============================================================================

@tool
def get_team_statistics(
    team: str,
    season: str = "2025",
    stat_category: str = "batting"
) -> Dict[str, Any]:
    """
    특정 팀의 시즌 통계를 조회합니다.
    
    Args:
        team: 팀명 (정규화된 영문명)
        season: 시즌 연도
        stat_category: 통계 카테고리 ('batting', 'pitching', 'fielding')
    
    Returns:
        팀 통계 데이터
    """
    # 실제 구현에서는 데이터베이스/API에서 조회
    # 현재는 플레이스홀더 반환
    return {
        "team": team,
        "season": season,
        "category": stat_category,
        "stats": {
            "message": f"{team}의 {season} 시즌 {stat_category} 통계를 조회합니다.",
            "data_source": "vector_store",
            "note": "실제 데이터는 ChromaDB에서 검색됩니다."
        }
    }


# =============================================================================
# 팀 비교 도구
# =============================================================================

@tool
def compare_teams(
    team1: str,
    team2: str,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    두 팀의 성적을 비교 분석합니다.
    
    Args:
        team1: 첫 번째 팀명
        team2: 두 번째 팀명
        metrics: 비교할 지표 목록
    
    Returns:
        팀 비교 결과
    """
    if metrics is None:
        metrics = ["승률", "타율", "평균자책점"]
    
    return {
        "comparison": {
            "team1": team1,
            "team2": team2,
            "metrics": metrics,
            "message": f"{team1}와 {team2}의 {', '.join(metrics)} 지표를 비교합니다.",
        }
    }


# =============================================================================
# 도구 목록
# =============================================================================

# Agent에서 사용할 도구 목록
AVAILABLE_TOOLS = [
    generate_dashboard_json,
    get_team_statistics,
    compare_teams,
]


def get_tools() -> List:
    """
    사용 가능한 도구 목록을 반환합니다.
    
    Returns:
        List: LangChain 도구 목록
    """
    return AVAILABLE_TOOLS


# =============================================================================
# OpenAI Function Calling 스키마
# =============================================================================

def get_openai_functions() -> List[Dict]:
    """
    OpenAI Function Calling용 함수 스키마를 반환합니다.
    
    Returns:
        List[Dict]: 함수 정의 목록
    """
    return [
        {
            "name": "generate_dashboard_json",
            "description": "프론트엔드용 대시보드 JSON을 생성합니다. 시각화가 필요한 분석 결과를 구조화된 형태로 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard_type": {
                        "type": "string",
                        "enum": ["season_analysis", "match_analysis", "team_comparison"],
                        "description": "대시보드 유형"
                    },
                    "teams": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "분석 대상 팀 목록 (영문명)"
                    },
                    "title": {
                        "type": "string",
                        "description": "대시보드 제목"
                    },
                    "date": {
                        "type": "string",
                        "description": "경기 날짜 (YYYY-MM-DD, match_analysis인 경우)"
                    },
                    "season": {
                        "type": "string",
                        "description": "시즌 연도",
                        "default": "2025"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "표시할 지표 목록"
                    }
                },
                "required": ["dashboard_type", "teams", "title"]
            }
        },
        {
            "name": "get_team_statistics",
            "description": "특정 팀의 시즌 통계를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "팀명 (영문명)"
                    },
                    "season": {
                        "type": "string",
                        "description": "시즌 연도",
                        "default": "2025"
                    },
                    "stat_category": {
                        "type": "string",
                        "enum": ["batting", "pitching", "fielding"],
                        "description": "통계 카테고리"
                    }
                },
                "required": ["team"]
            }
        },
        {
            "name": "compare_teams",
            "description": "두 팀의 성적을 비교 분석합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team1": {
                        "type": "string",
                        "description": "첫 번째 팀명"
                    },
                    "team2": {
                        "type": "string",
                        "description": "두 번째 팀명"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "비교할 지표 목록"
                    }
                },
                "required": ["team1", "team2"]
            }
        }
    ]


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":
    # 대시보드 생성 테스트
    print("=" * 60)
    print("대시보드 생성 테스트")
    print("=" * 60)
    
    # 시즌 분석 대시보드
    season_dashboard = generate_dashboard_json.invoke({
        "dashboard_type": "season_analysis",
        "teams": ["Hanwha", "LG"],
        "title": "2025 시즌 한화 vs LG 분석",
        "season": "2025",
        "metrics": ["타율", "홈런", "ERA", "승률"]
    })
    
    print("\n시즌 분석 대시보드:")
    print(json.dumps(season_dashboard, ensure_ascii=False, indent=2)[:500])
    
    # 경기 분석 대시보드
    match_dashboard = generate_dashboard_json.invoke({
        "dashboard_type": "match_analysis",
        "teams": ["Hanwha", "LG"],
        "title": "한화 vs LG 경기 분석",
        "date": "2025-05-01"
    })
    
    print("\n경기 분석 대시보드:")
    print(json.dumps(match_dashboard, ensure_ascii=False, indent=2)[:500])
