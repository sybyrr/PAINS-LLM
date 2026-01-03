"""
Utils module - 전처리 및 정규화 로직

RapidFuzz를 사용한 팀/선수명 정규화 및 유틸리티 함수를 제공합니다.
논문의 Query Cleaning and Normalization (Section 4.3.2) 개선 버전입니다.
"""

from rapidfuzz import fuzz, process
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import re
import os

from .config import TEAM_MATCH_THRESHOLD, PLAYER_MATCH_THRESHOLD

# 추가 라이브러리
from openai import OpenAI
from typing import List, Dict, Any
from datetime import datetime


# =============================================================================
# 선수 이름 영어 변환 (캐싱 지원)
# =============================================================================

_player_name_cache: Dict[str, str] = {}
_openai_client: OpenAI = None


def get_openai_client() -> OpenAI:
    """OpenAI 클라이언트 싱글톤"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def romanize_player_name(korean_name: str) -> str:
    """
    한국어 선수 이름을 로마자(영어)로 변환합니다.
    
    캐시를 사용하여 동일한 이름의 중복 번역을 방지합니다.
    
    Args:
        korean_name: 한국어 선수 이름 (예: "류현진", "후라도")
    
    Returns:
        str: 로마자 이름 (예: "Ryu Hyun-jin", "Hueraldo")
    """
    global _player_name_cache
    
    # 이미 영어인 경우 그대로 반환
    if korean_name and all(ord(c) < 128 or c.isspace() for c in korean_name):
        return korean_name
    
    # 캐시 확인
    if korean_name in _player_name_cache:
        return _player_name_cache[korean_name]
    
    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Korean-to-English name romanizer for KBO baseball players. "
                        "Convert Korean player names to their official romanized form. "
                        "For foreign players, use their original name (e.g., 후라도 → Hueraldo). "
                        "For Korean players, use standard romanization (e.g., 류현진 → Ryu Hyun-jin). "
                        "Output ONLY the romanized name, nothing else."
                    )
                },
                {"role": "user", "content": korean_name}
            ],
            temperature=0
        )
        
        romanized = response.choices[0].message.content.strip()
        _player_name_cache[korean_name] = romanized
        return romanized
        
    except Exception as e:
        # API 오류 시 원본 이름 반환
        print(f"⚠️ 이름 변환 실패 ({korean_name}): {e}")
        return korean_name


def romanize_player_names_batch(names: List[str]) -> Dict[str, str]:
    """
    여러 선수 이름을 한 번에 로마자로 변환합니다.
    배치 처리로 API 호출 횟수를 줄입니다.
    
    Args:
        names: 한국어 선수 이름 리스트
    
    Returns:
        Dict[str, str]: 한국어 이름 -> 로마자 이름 매핑
    """
    global _player_name_cache
    
    # 이미 캐시된 이름 제외
    uncached_names = [n for n in names if n not in _player_name_cache and n]
    
    if not uncached_names:
        return {n: _player_name_cache.get(n, n) for n in names}
    
    # 영어 이름은 그대로 유지
    korean_names = []
    for name in uncached_names:
        if all(ord(c) < 128 or c.isspace() for c in name):
            _player_name_cache[name] = name
        else:
            korean_names.append(name)
    
    if not korean_names:
        return {n: _player_name_cache.get(n, n) for n in names}
    
    try:
        client = get_openai_client()
        
        names_text = "\n".join(korean_names)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Korean-to-English name romanizer for KBO baseball players. "
                        "Convert each Korean player name to their official romanized form. "
                        "For foreign players, use their original name (e.g., 후라도 → Hueraldo). "
                        "For Korean players, use standard romanization (e.g., 류현진 → Ryu Hyun-jin). "
                        "Output one romanized name per line, in the same order as input. "
                        "Output ONLY the names, no numbers or explanations."
                    )
                },
                {"role": "user", "content": names_text}
            ],
            temperature=0
        )
        
        romanized_names = response.choices[0].message.content.strip().split("\n")
        
        for korean, romanized in zip(korean_names, romanized_names):
            _player_name_cache[korean] = romanized.strip()
        
    except Exception as e:
        print(f"⚠️ 배치 이름 변환 실패: {e}")
        for name in korean_names:
            _player_name_cache[name] = name
    
    return {n: _player_name_cache.get(n, n) for n in names}


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
    "꼴데" : "Lotte",
    
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
    "쓱": "SSG",
    
    # 키움 히어로즈
    "키움": "Kiwoom",
    "키움 히어로즈": "Kiwoom",
    "키움히어로즈": "Kiwoom",
    "히어로즈": "Kiwoom",
    "Kiwoom Heroes": "Kiwoom",
    "kiwoom": "Kiwoom",
    "넥센" : "Kiwoom",
    
    # KT 위즈
    "KT": "KT",
    "케이티": "KT",
    "KT 위즈": "KT",
    "KT위즈": "KT",
    "위즈": "KT",
    "KT Wiz": "KT",
    "kt": "KT",
    "크트": "KT"
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


def clean_query_for_embedding(
    query: str, 
    teams: List[str] = None, 
    date: str = None,
    query_type: str = None
) -> str:
    """
    임베딩 검색을 위해 쿼리를 정제합니다.
    
    논문 Section 4.3.2의 "Fully Cleaned" 쿼리 전략을 구현합니다.
    - 원본 쿼리에서 핵심 엔티티(선수명, 팀명, 시즌)를 추출
    - 정해진 형식으로 검색 쿼리를 구성
    
    Args:
        query: 원본 사용자 쿼리
        teams: 정규화된 팀 목록
        date: 추출된 날짜
        query_type: 쿼리 유형 ("season_analysis" 또는 "match_analysis")
    
    Returns:
        str: 정규화된 검색 쿼리
    """
    from .retriever import translate_query_to_english
    
    # 1. 원본 쿼리를 영어로 번역 (선수명, 시즌 정보 추출을 위해)
    translated = translate_query_to_english(query)
    
    # 2. 쿼리 유형에 따른 정규화된 형식 구성
    if query_type == "match_analysis":
        # 경기 분석: "Find dataset for game between {team1} and {team2} on {date}. {details}"
        # 선수 정보 등 추가 컨텍스트를 위해 번역된 쿼리도 함께 포함
        if teams and len(teams) >= 2:
            base = f"Find dataset for game between {teams[0]} and {teams[1]}"
            if date:
                base += f" on {date}"
            # 번역된 쿼리에 선수 정보 등 추가 정보가 있을 수 있으므로 포함
            return f"{base}. {translated}"
        elif date:
            return f"Find dataset for game on {date}. {translated}"
        else:
            return translated
    
    elif query_type == "season_analysis":
        # 시즌 분석: 선수명이 포함된 경우와 팀 분석의 경우를 구분
        if teams:
            # 번역된 쿼리에 팀명이 아닌 다른 정보(선수명 등)가 있는지 확인
            # translated에서 팀명을 제외한 나머지가 선수 정보
            return f"Find dataset for {teams[0]} player season stats. {translated}"
        else:
            return f"Find dataset for player season stats. {translated}"
    
    else:
        # 기본: 번역된 쿼리 사용
        return translated


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
        Season: "Ryu Hyun-jin 2025 Regular season pitching stats. Team: Hanwha. ERA: 3.23..."
        Match: "Pitcher Koo Chang-mo from NC on 2025-10-06 (Post). Result: W, IP: 6, ER: 1, SO: 5"
    """
    if data_type == "season":
        # 시즌 데이터용 설명 문장 (선수별 누적 통계)
        team = data.get("Team", data.get("team", "Unknown Team"))
        season = data.get("season", "2025")
        season_type = data.get("_season_type", "Regular")
        stat_type = data.get("_stat_type", "pitching")
        player_name_kr = data.get("Name", "Unknown Player")
        
        # 선수 이름을 영어로 변환
        player_name = romanize_player_name(player_name_kr)
        
        # 투수 통계 주요 지표
        if stat_type == "pitching":
            era = data.get("ERA", "N/A")
            wins = data.get("W", 0)
            losses = data.get("L", 0)
            ip = data.get("IP", 0)
            so = data.get("SO", 0)
            
            # 선수 이름을 맨 앞에 배치하여 검색 유사도 향상
            # 쿼리: "Ryu Hyun-jin 2025 season stats" → Description: "Ryu Hyun-jin 2025 Regular season..."
            return (
                f"{player_name} {season} {season_type} season pitching stats. "
                f"Team: {team}. ERA: {era}, W: {wins}, L: {losses}, IP: {ip}, SO: {so}. "
                f"KBO baseball pitcher season statistics."
            )
        else:
            # 타자 통계 (향후 확장)
            avg = data.get("AVG", "N/A")
            hr = data.get("HR", 0)
            rbi = data.get("RBI", 0)
            
            return (
                f"{player_name} {season} {season_type} season batting stats. "
                f"Team: {team}. AVG: {avg}, HR: {hr}, RBI: {rbi}. "
                f"KBO baseball batter season statistics."
            )
    
    elif data_type == "match":
        # 경기별 투수/타자 기록용 설명 문장 (날짜+팀으로 그룹화된 데이터)
        team = data.get("Team", "Unknown Team")
        date = data.get("Date", data.get("date", "Unknown date"))
        season_type = data.get("_season_type", "Regular")
        year = data.get("_year", "2025")
        record_type = data.get("_record_type", "pitcher")
        players = data.get("players", [])
        
        if record_type == "pitcher":
            # 투수 기록 요약 생성
            player_summaries = []
            total_ip = 0
            total_er = 0
            total_so = 0
            game_result = None
            
            for p in players:
                name = p.get("Name", "Unknown")
                result = p.get("Result", "")
                ip = p.get("IP", 0)
                er = p.get("ER", 0)
                so = p.get("SO", 0)
                
                total_ip += ip if isinstance(ip, (int, float)) else 0
                total_er += er if isinstance(er, (int, float)) else 0
                total_so += so if isinstance(so, (int, float)) else 0
                
                # 경기 결과 (승/패/세이브 등) 추출
                if result in ["승", "패", "세", "홀드"]:
                    player_summaries.append(f"{name}({result})")
                    if result in ["승", "패"]:
                        game_result = result
                else:
                    player_summaries.append(name)
            
            players_text = ", ".join(player_summaries) if player_summaries else "No pitchers"
            result_text = f" Game result: {game_result}." if game_result else ""
            
            return (
                f"{team} pitching data on {date} ({year} {season_type}).{result_text} "
                f"Pitchers: {players_text}. "
                f"Team totals - IP: {total_ip}, ER: {total_er}, SO: {total_so}. "
                f"KBO baseball team pitching game record."
            )
        else:
            # 타자 기록인 경우 (향후 확장용)
            player_names = [p.get("Name", "Unknown") for p in players]
            players_text = ", ".join(player_names) if player_names else "No batters"
            
            return (
                f"{team} batting lineup on {date} ({year} {season_type}). "
                f"Batters: {players_text}. "
                f"KBO baseball team batting game record."
            )
    
    return "KBO Baseball data."

### ====== generate_game_description 함수 =====

def summarize_team_pitchers(pitchers: List[Dict[str, Any]], team_name: str) -> str:
    """
    팀 투수진 기록을 요약합니다.
    
    NOTE: 선수 이름은 포함하지 않습니다.
    이유: "류현진 시즌 성적" 같은 쿼리가 match 데이터를 검색하는 것을 방지.
    선수 이름은 season 데이터에서만 검색되어야 합니다.
    
    Args:
        pitchers: 투수 레코드 리스트
        team_name: 팀 이름
    
    Returns:
        str: 투수진 요약 문자열 (팀 통계만 포함, 선수명 제외)
    """
    if not pitchers:
        return f"{team_name} 투수진: 기록 없음"
    
    total_ip = 0.0
    total_er = 0
    total_so = 0
    win_count = 0
    loss_count = 0
    
    for p in pitchers:
        try:
            ip = float(p.get('IP', 0))
            er = int(p.get('ER', 0))
            so = int(p.get('SO', 0))
            total_ip += ip
            total_er += er
            total_so += so
        except (ValueError, TypeError):
            pass
        
        result = p.get('Result', '')
        if result == '승':
            win_count += 1
        elif result == '패':
            loss_count += 1
    
    pitcher_count = len(pitchers)
    
    # 선수명 없이 팀 통계만 포함
    summary = f"{team_name} 투수진: {pitcher_count}명 등판, {total_ip:.0f}이닝, {total_er}자책점, {total_so}삼진"
    
    if win_count > 0 or loss_count > 0:
        summary += f" (승{win_count}/패{loss_count})"
    
    return summary


def generate_game_description(game_data: Dict[str, Any]) -> str:
    """
    경기 단위 문서의 설명 문장을 생성합니다.
    양 팀 투수 기록을 하나의 문장으로 요약합니다.
    
    Args:
        game_data: 경기별 그룹화된 데이터
            - date, home_team, away_team, home_runs, away_runs
            - home_pitchers: 홈팀 투수 리스트
            - away_pitchers: 원정팀 투수 리스트
    
    Returns:
        str: 임베딩용 설명 문장
    
    Example:
        "2025년 04월 17일 Regular 시즌 경기: LG(홈) vs Lotte(원정). 
         스코어 3-7 (Lotte 승리). 
         LG 투수진(8이닝 7자책): 이민호(패), 김진성, 고우석. 
         Lotte 투수진(9이닝 3자책): 반즈(승), 송재영, 박준우."
    """
    date = game_data.get('date', '')
    home = game_data.get('home_team', 'Unknown')
    away = game_data.get('away_team', 'Unknown')
    home_runs = game_data.get('home_runs', '?')
    away_runs = game_data.get('away_runs', '?')
    season_type = game_data.get('season_type', 'Regular')
    
    # 날짜 포맷
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        date_str = dt.strftime("%Y년 %m월 %d일")
    except ValueError:
        date_str = date
    
    # 승패 결정
    result = ""
    try:
        home_score = int(home_runs)
        away_score = int(away_runs)
        if home_score > away_score:
            result = f"{home} 승리"
        elif away_score > home_score:
            result = f"{away} 승리"
        else:
            result = "무승부"
    except (ValueError, TypeError):
        pass
    
    # 투수 요약 생성
    home_summary = summarize_team_pitchers(
        game_data.get('home_pitchers', []), home
    )
    away_summary = summarize_team_pitchers(
        game_data.get('away_pitchers', []), away
    )
    
    # 최종 설명 문장
    description = (
        f"{date_str} {season_type} 시즌 경기: {home}(홈) vs {away}(원정). "
        f"스코어 {home_runs}-{away_runs}"
    )
    
    if result:
        description += f" ({result}). "
    else:
        description += ". "
    
    description += f"{home_summary}. {away_summary}."
    
    return description
