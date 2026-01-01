"""
Utils module - 전처리 및 정규화 로직

RapidFuzz를 사용한 팀/선수명 정규화 및 유틸리티 함수를 제공합니다.
논문의 Query Cleaning and Normalization (Section 4.3.2) 개선 버전입니다.
"""

from rapidfuzz import fuzz, process
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import re

from .config import TEAM_MATCH_THRESHOLD, PLAYER_MATCH_THRESHOLD

# =============================================================================
# KBO 팀 매핑 딕셔너리
# 다양한 팀 명칭(별명, 약어, 영문명 등)을 공식 명칭으로 매핑
# =============================================================================
TEAM_MAP: Dict[str, str] = {
    # 한화 이글스
    "한화": "Hanwha",
    "한화 이글스": "Hanwha",
    "한화이글스": "Hanwha",
    "이글스": "Hanwha",
    "Hanwha Eagles": "Hanwha",
    "hanwha": "Hanwha",
    
    # LG 트윈스
    "LG": "LG",
    "엘지": "LG",
    "LG 트윈스": "LG",
    "LG트윈스": "LG",
    "트윈스": "LG",
    "엘지트윈스": "LG",
    "LG Twins": "LG",
    
    # 삼성 라이온즈
    "삼성": "Samsung",
    "삼성 라이온즈": "Samsung",
    "삼성라이온즈": "Samsung",
    "라이온즈": "Samsung",
    "Samsung Lions": "Samsung",
    "samsung": "Samsung",
    
    # 두산 베어스
    "두산": "Doosan",
    "두산 베어스": "Doosan",
    "두산베어스": "Doosan",
    "베어스": "Doosan",
    "Doosan Bears": "Doosan",
    "doosan": "Doosan",
    
    # 롯데 자이언츠
    "롯데": "Lotte",
    "롯데 자이언츠": "Lotte",
    "롯데자이언츠": "Lotte",
    "자이언츠": "Lotte",
    "Lotte Giants": "Lotte",
    "lotte": "Lotte",
    
    # 기아 타이거즈
    "기아": "KIA",
    "KIA": "KIA",
    "기아 타이거즈": "KIA",
    "기아타이거즈": "KIA",
    "타이거즈": "KIA",
    "KIA Tigers": "KIA",
    "kia": "KIA",
    
    # NC 다이노스
    "NC": "NC",
    "엔씨": "NC",
    "NC 다이노스": "NC",
    "NC다이노스": "NC",
    "다이노스": "NC",
    "NC Dinos": "NC",
    "nc": "NC",
    
    # SSG 랜더스
    "SSG": "SSG",
    "에스에스지": "SSG",
    "SSG 랜더스": "SSG",
    "SSG랜더스": "SSG",
    "랜더스": "SSG",
    "인천": "SSG",
    "SSG Landers": "SSG",
    "ssg": "SSG",
    
    # 키움 히어로즈
    "키움": "Kiwoom",
    "키움 히어로즈": "Kiwoom",
    "키움히어로즈": "Kiwoom",
    "히어로즈": "Kiwoom",
    "Kiwoom Heroes": "Kiwoom",
    "kiwoom": "Kiwoom",
    
    # KT 위즈
    "KT": "KT",
    "케이티": "KT",
    "KT 위즈": "KT",
    "KT위즈": "KT",
    "위즈": "KT",
    "KT Wiz": "KT",
    "kt": "KT",
}

# 공식 팀 목록 (영문)
OFFICIAL_TEAMS: List[str] = [
    "Hanwha", "LG", "Samsung", "Doosan", "Lotte", 
    "KIA", "NC", "SSG", "Kiwoom", "KT"
]

# 팀별 한글-영문 매핑 (역방향)
TEAM_KO_TO_EN: Dict[str, str] = {
    "한화": "Hanwha",
    "LG": "LG",
    "삼성": "Samsung",
    "두산": "Doosan",
    "롯데": "Lotte",
    "기아": "KIA",
    "NC": "NC",
    "SSG": "SSG",
    "키움": "Kiwoom",
    "KT": "KT",
}

TEAM_EN_TO_KO: Dict[str, str] = {v: k for k, v in TEAM_KO_TO_EN.items()}


# =============================================================================
# 정규화 함수
# =============================================================================

def normalize_team_name(team_input: str) -> Tuple[Optional[str], float]:
    """
    사용자 입력에서 팀명을 공식 명칭으로 정규화합니다.
    
    RapidFuzz의 QRatio scorer를 사용하여 퍼지 매칭을 수행합니다.
    논문에서 검증된 최적 scorer입니다.
    
    Args:
        team_input: 사용자가 입력한 팀명 (예: "한화", "이글스", "hanwha eagles")
    
    Returns:
        Tuple[Optional[str], float]: (정규화된 팀명, 신뢰도 점수)
        매칭 실패시 (None, 0.0) 반환
    
    Example:
        >>> normalize_team_name("한화 어제 경기")
        ("Hanwha", 95.0)
    """
    if not team_input:
        return None, 0.0
    
    # 입력 전처리: 소문자화, 공백 정리
    cleaned_input = team_input.strip().lower()
    
    # 1단계: 정확한 매칭 시도 (가장 빠름)
    if team_input in TEAM_MAP:
        return TEAM_MAP[team_input], 100.0
    
    # 2단계: RapidFuzz를 사용한 퍼지 매칭
    # process.extractOne: 최고 매칭 결과 하나만 반환
    result = process.extractOne(
        query=cleaned_input,
        choices=list(TEAM_MAP.keys()),
        scorer=fuzz.QRatio,  # 논문에서 검증된 scorer
        score_cutoff=TEAM_MATCH_THRESHOLD
    )
    
    if result:
        matched_key, score, _ = result
        canonical_team = TEAM_MAP[matched_key]
        return canonical_team, score
    
    return None, 0.0


def normalize_player_name(player_input: str, player_list: List[str]) -> Tuple[Optional[str], float]:
    """
    사용자 입력에서 선수명을 정규화합니다.
    
    Args:
        player_input: 사용자가 입력한 선수명
        player_list: 매칭할 선수 목록 (데이터에서 추출)
    
    Returns:
        Tuple[Optional[str], float]: (정규화된 선수명, 신뢰도 점수)
    """
    if not player_input or not player_list:
        return None, 0.0
    
    result = process.extractOne(
        query=player_input.strip(),
        choices=player_list,
        scorer=fuzz.QRatio,
        score_cutoff=PLAYER_MATCH_THRESHOLD
    )
    
    if result:
        matched_name, score, _ = result
        return matched_name, score
    
    return None, 0.0


def extract_teams_from_query(query: str) -> List[Tuple[str, float]]:
    """
    쿼리에서 모든 팀명을 추출하고 정규화합니다.
    
    Args:
        query: 사용자 쿼리
    
    Returns:
        List[Tuple[str, float]]: [(정규화된 팀명, 신뢰도), ...]
    """
    found_teams = []
    query_lower = query.lower()
    
    # 각 팀 키워드에 대해 쿼리에 포함되어 있는지 확인
    for key, canonical in TEAM_MAP.items():
        key_lower = key.lower()
        if key_lower in query_lower:
            # 중복 방지
            if canonical not in [t[0] for t in found_teams]:
                found_teams.append((canonical, 100.0))
    
    # 팀을 찾지 못한 경우, 퍼지 매칭으로 단어별 검색
    if not found_teams:
        words = query.split()
        for word in words:
            if len(word) >= 2:  # 최소 2글자 이상
                team, score = normalize_team_name(word)
                if team and team not in [t[0] for t in found_teams]:
                    found_teams.append((team, score))
    
    return found_teams


def extract_date_from_query(query: str) -> Optional[str]:
    """
    쿼리에서 날짜 정보를 추출합니다.
    
    Args:
        query: 사용자 쿼리
    
    Returns:
        Optional[str]: YYYY-MM-DD 형식의 날짜 문자열
    """
    # 패턴 1: YYYY-MM-DD 또는 YYYY.MM.DD
    pattern_full = r'(\d{4})[-./](\d{1,2})[-./](\d{1,2})'
    match = re.search(pattern_full, query)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    
    # 패턴 2: MM월 DD일
    pattern_korean = r'(\d{1,2})월\s*(\d{1,2})일'
    match = re.search(pattern_korean, query)
    if match:
        month, day = match.groups()
        year = datetime.now().year
        return f"{year}-{int(month):02d}-{int(day):02d}"
    
    # 패턴 3: 자연어 날짜
    today = datetime.now()
    if "오늘" in query:
        return today.strftime("%Y-%m-%d")
    elif "어제" in query:
        from datetime import timedelta
        yesterday = today - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")
    elif "그제" in query or "그저께" in query:
        from datetime import timedelta
        day_before = today - timedelta(days=2)
        return day_before.strftime("%Y-%m-%d")
    
    return None


def clean_query_for_embedding(query: str, teams: List[str] = None, date: str = None) -> str:
    """
    임베딩 검색을 위해 쿼리를 정제합니다.
    
    논문의 Fully Cleaned 쿼리 전략을 구현합니다.
    불필요한 단어를 제거하고 핵심 엔티티만 남깁니다.
    
    Args:
        query: 원본 사용자 쿼리
        teams: 정규화된 팀 목록
        date: 추출된 날짜
    
    Returns:
        str: 정제된 검색 쿼리
    """
    # 검색에 최적화된 쿼리 구성
    parts = []
    
    if teams:
        if len(teams) >= 2:
            # 두 팀이 있으면 경기 분석으로 추정
            parts.append(f"Game between {teams[0]} and {teams[1]}")
        else:
            parts.append(f"Data for team {teams[0]}")
    
    if date:
        parts.append(f"on {date}")
    
    if not parts:
        # 팀/날짜 정보가 없으면 원본 쿼리 일부 사용
        return query
    
    return " ".join(parts)


def build_metadata_filter(
    teams: List[str] = None,
    data_type: str = None,
    date: str = None,
    season: str = None
) -> Dict:
    """
    ChromaDB 메타데이터 필터를 생성합니다.
    
    Args:
        teams: 필터링할 팀 목록
        data_type: 데이터 유형 ("season" 또는 "match")
        date: 경기 날짜
        season: 시즌 (예: "2025")
    
    Returns:
        Dict: ChromaDB where 필터
    """
    conditions = []
    
    if data_type:
        conditions.append({"type": data_type})
    
    if teams:
        # $in 연산자: 팀 목록 중 하나라도 포함된 경우
        conditions.append({"teams": {"$in": teams}})
    
    if date:
        conditions.append({"date": date})
    
    if season:
        conditions.append({"season": season})
    
    if not conditions:
        return {}
    
    if len(conditions) == 1:
        return conditions[0]
    
    # 여러 조건을 AND로 결합
    return {"$and": conditions}


# =============================================================================
# 데이터 전처리 함수
# =============================================================================

def generate_descriptive_sentence(data: dict, data_type: str) -> str:
    """
    JSON 데이터에서 임베딩용 설명 문장을 생성합니다.
    
    논문 Section 4.2의 핵심 전략:
    JSON을 그대로 임베딩하지 않고, 자연어 설명 문장을 생성하여 임베딩합니다.
    
    Args:
        data: JSON 데이터
        data_type: "season" 또는 "match"
    
    Returns:
        str: 임베딩용 설명 문장
    
    Example:
        Season: "2025 Regular season pitching stats for 류현진 (NC). ERA: 2.85, W: 12, L: 5"
        Match: "Pitcher 구창모 from NC on 2025-10-06 (Post). Result: 승, IP: 6, ER: 1, SO: 5"
    """
    if data_type == "season":
        # 시즌 데이터용 설명 문장 (선수별 누적 통계)
        team = data.get("Team", data.get("team", "Unknown Team"))
        season = data.get("season", "2025")
        season_type = data.get("_season_type", "Regular")
        stat_type = data.get("_stat_type", "pitching")
        player_name = data.get("Name", "Unknown Player")
        
        # 투수 통계 주요 지표
        if stat_type == "pitching":
            era = data.get("ERA", "N/A")
            wins = data.get("W", 0)
            losses = data.get("L", 0)
            ip = data.get("IP", 0)
            so = data.get("SO", 0)
            
            return (
                f"{season} {season_type} season pitching stats for {player_name} ({team}). "
                f"ERA: {era}, W: {wins}, L: {losses}, IP: {ip}, SO: {so}. "
                f"KBO baseball pitcher season statistics."
            )
        else:
            # 타자 통계 (향후 확장)
            avg = data.get("AVG", "N/A")
            hr = data.get("HR", 0)
            rbi = data.get("RBI", 0)
            
            return (
                f"{season} {season_type} season batting stats for {player_name} ({team}). "
                f"AVG: {avg}, HR: {hr}, RBI: {rbi}. "
                f"KBO baseball batter season statistics."
            )
    
    elif data_type == "match":
        # 경기별 투수/타자 기록용 설명 문장
        team = data.get("Team", "Unknown Team")
        player_name = data.get("Name", "Unknown Player")
        date = data.get("Date", data.get("date", "Unknown date"))
        season_type = data.get("_season_type", "Regular")
        year = data.get("_year", "2025")
        record_type = data.get("_record_type", "pitcher")
        
        if record_type == "pitcher":
            result = data.get("Result", "")
            ip = data.get("IP", 0)
            er = data.get("ER", 0)
            so = data.get("SO", 0)
            bb_hp = data.get("BB_HP", 0)
            
            result_text = f", Result: {result}" if result else ""
            
            return (
                f"Pitcher {player_name} from {team} on {date} ({year} {season_type}). "
                f"IP: {ip}, ER: {er}, SO: {so}, BB+HP: {bb_hp}{result_text}. "
                f"KBO baseball pitcher game performance record."
            )
        else:
            # 타자 기록인 경우 (향후 확장용)
            return (
                f"Batter {player_name} from {team} on {date} ({year} {season_type}). "
                f"KBO baseball batter game performance record."
            )
    
    return "KBO Baseball data."
