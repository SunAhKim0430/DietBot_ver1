from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI
from supabase import Client, create_client

APP_TITLE = "당뇨 식단 보조 비서"

EMERGENCY_LOW = 70
HIGH_WARNING = 250
VERY_HIGH_WARNING = 300

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_USER_ID = "dad_001"

SYSTEM_GUIDE = """
이 앱은 당뇨 및 체중 관리 보조용입니다.

- 진단이나 처방은 하지 않습니다.
- 혈당, 식사, 운동, 약 복용, 증상 기록을 도와줍니다.
- 냉장고 식재료를 바탕으로 식단 아이디어를 제안합니다.
- 응급 상황 판단을 대신하지 않습니다.
"""

DISCLAIMER = """
주의:
이 앱은 의료진을 대체하지 않습니다.
심한 저혈당 증상, 의식 저하, 호흡 이상, 지속적인 고혈당이 있으면 즉시 의료기관에 연락하세요.
"""


# =========================
# 클라이언트
# =========================
def get_supabase_client() -> Client | None:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


def get_openai_client() -> OpenAI | None:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        return None


# =========================
# 세션 초기화
# =========================
def init_session_state() -> None:
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "user_id" not in st.session_state:
        st.session_state.user_id = DEFAULT_USER_ID

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요. 혈당 기록과 식단 추천을 도와드립니다.\n"
                    "냉장고 식재료를 저장해두면, 그 재료 위주로 식단을 추천해드릴 수 있습니다."
                ),
            }
        ]

    if "logs_df" not in st.session_state:
        st.session_state.logs_df = pd.DataFrame(
            columns=["timestamp", "entry_type", "blood_sugar", "message", "response"]
        )

    if "ingredients" not in st.session_state:
        st.session_state.ingredients = []

    if "profile" not in st.session_state:
        st.session_state.profile = {
            "goal": "당뇨 관리 + 체중 감량",
            "avoid_foods": "",
            "notes": "",
        }


# =========================
# DB 읽기/쓰기
# =========================
def load_profile_from_db() -> None:
    supabase = get_supabase_client()
    if supabase is None:
        return

    try:
        response = (
            supabase.table("user_profiles")
            .select("*")
            .eq("user_id", st.session_state.user_id)
            .limit(1)
            .execute()
        )

        rows = response.data or []
        if not rows:
            return

        row = rows[0]
        st.session_state.ingredients = row.get("ingredients") or []
        st.session_state.profile = {
            "goal": row.get("goal") or "당뇨 관리 + 체중 감량",
            "avoid_foods": row.get("avoid_foods") or "",
            "notes": row.get("notes") or "",
        }
    except Exception as e:
        st.error(f"프로필 불러오기 실패: {e}")


def save_profile_to_db() -> tuple[bool, str]:
    supabase = get_supabase_client()
    if supabase is None:
        return False, "Supabase 설정이 없습니다."

    payload = {
        "user_id": st.session_state.user_id,
        "ingredients": st.session_state.ingredients,
        "goal": st.session_state.profile["goal"],
        "avoid_foods": st.session_state.profile["avoid_foods"],
        "notes": st.session_state.profile["notes"],
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        (
            supabase.table("user_profiles")
            .upsert(payload)
            .execute()
        )
        return True, "저장되었습니다."
    except Exception as e:
        return False, f"저장 실패: {e}"


def load_logs_from_db(limit: int = 50) -> None:
    supabase = get_supabase_client()
    if supabase is None:
        return

    try:
        response = (
            supabase.table("health_logs")
            .select("timestamp, entry_type, blood_sugar, message, response")
            .eq("user_id", st.session_state.user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        rows = response.data or []
        if rows:
            df = pd.DataFrame(rows)
            st.session_state.logs_df = df
        else:
            st.session_state.logs_df = pd.DataFrame(
                columns=["timestamp", "entry_type", "blood_sugar", "message", "response"]
            )
    except Exception as e:
        st.error(f"기록 불러오기 실패: {e}")


def insert_log_to_db(entry_type: str, blood_sugar: int | None, message: str, response: str) -> None:
    supabase = get_supabase_client()
    if supabase is None:
        return

    payload = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "entry_type": entry_type,
        "blood_sugar": blood_sugar,
        "message": message,
        "response": response,
    }

    try:
        supabase.table("health_logs").insert(payload).execute()
    except Exception as e:
        st.error(f"기록 저장 실패: {e}")


# =========================
# 유틸
# =========================
def extract_blood_sugar(text: str) -> int | None:
    numbers = re.findall(r"\d{2,3}", text)
    if not numbers:
        return None

    for num in numbers:
        try:
            value = int(num)
            if 20 <= value <= 600:
                return value
        except ValueError:
            continue
    return None


def normalize_ingredients(raw_text: str) -> list[str]:
    parts = re.split(r"[,/\n]+", raw_text)
    cleaned = []
    seen = set()

    for item in parts:
        value = item.strip()
        if not value:
            continue
        if value not in seen:
            cleaned.append(value)
            seen.add(value)

    return cleaned


def ingredients_to_text() -> str:
    if not st.session_state.ingredients:
        return "없음"
    return ", ".join(st.session_state.ingredients)


def is_meal_plan_request(text: str) -> bool:
    keywords = [
        "식단", "메뉴", "추천", "뭐 먹", "오늘 뭐 먹",
        "아침 추천", "점심 추천", "저녁 추천",
        "식사 추천", "당뇨식", "다이어트 식단"
    ]
    return any(k in text for k in keywords)


# =========================
# 일반 응답
# =========================
def generate_basic_response(user_text: str) -> tuple[str, int | None, str]:
    blood_sugar = extract_blood_sugar(user_text)

    if blood_sugar is not None:
        if blood_sugar < EMERGENCY_LOW:
            response = (
                f"현재 혈당 {blood_sugar} mg/dL는 저혈당 범위입니다.\n\n"
                "- 가능하면 즉시 당분(주스, 사탕 등)을 섭취하세요.\n"
                "- 15분 뒤 다시 혈당을 확인하세요.\n"
                "- 심한 어지럼, 식은땀, 떨림, 의식 저하가 있으면 즉시 보호자나 응급실에 연락하세요."
            )
        elif blood_sugar >= VERY_HIGH_WARNING:
            response = (
                f"현재 혈당 {blood_sugar} mg/dL는 매우 높은 편입니다.\n\n"
                "- 물을 충분히 드세요.\n"
                "- 처방받은 약/인슐린 복용 여부를 확인하세요.\n"
                "- 구토, 복통, 호흡 이상, 의식 저하가 있으면 즉시 병원에 연락하세요."
            )
        elif blood_sugar >= HIGH_WARNING:
            response = (
                f"현재 혈당 {blood_sugar} mg/dL는 높은 편입니다.\n\n"
                "- 최근 식사/간식 내용을 기록해 두세요.\n"
                "- 반복되면 병원 상담을 권합니다."
            )
        else:
            response = (
                f"현재 혈당 {blood_sugar} mg/dL가 기록되었습니다.\n\n"
                "- 식전/식후인지 함께 적어두면 더 좋습니다.\n"
                "- 필요하면 식단 추천도 해드릴게요."
            )

        return response, blood_sugar, "blood_sugar"

    if any(k in user_text for k in ["운동", "걷기", "산책"]):
        return (
            "운동 내용이 기록되었습니다.\n\n"
            "- 운동 전후 몸 상태를 확인하세요.\n"
            "- 어지럽거나 식은땀이 나면 바로 쉬고 혈당을 확인하세요."
        ), None, "exercise"

    if any(k in user_text for k in ["약", "인슐린", "복용"]):
        return (
            "약 복용/인슐린 관련 내용이 기록되었습니다.\n\n"
            "- 처방량은 임의로 바꾸지 마세요.\n"
            "- 복용 시간과 혈당을 함께 기록하면 관리에 도움이 됩니다."
        ), None, "medication"

    if any(k in user_text for k in ["식사", "밥", "죽", "아침", "점심", "저녁", "간식", "먹었"]):
        return (
            "식사 내용이 기록되었습니다.\n\n"
            "- 무엇을 얼마나 드셨는지 적어두면 더 좋습니다.\n"
            "- 식후 1~2시간 뒤 혈당 확인도 도움이 됩니다."
        ), None, "meal"

    if any(k in user_text for k in ["어지러", "식은땀", "떨", "구토", "기운이 없"]):
        return (
            "증상이 기록되었습니다.\n\n"
            "- 혈당을 먼저 재보세요.\n"
            "- 증상이 심하면 즉시 보호자 또는 의료기관에 연락하세요."
        ), None, "symptom"

    return (
        "기록할 내용을 말씀해 주세요.\n\n"
        "예시:\n"
        "- 혈당 125\n"
        "- 점심 먹었어\n"
        "- 20분 걷기\n"
        "- 약 복용함\n"
        "- 저녁 식단 추천해줘"
    ), None, "general"


# =========================
# OpenAI 식단 추천
# =========================
def build_meal_plan_prompt(user_request: str) -> str:
    profile = st.session_state.profile
    ingredients = st.session_state.ingredients

    return f"""
당신은 당뇨 및 체중 관리 보조 식단 코치입니다.
의료 진단이나 처방은 하지 말고, 일반적인 식사 아이디어만 제안하세요.

사용자 정보:
- 목표: {profile["goal"]}
- 피하고 싶은 음식: {profile["avoid_foods"] or "없음"}
- 추가 메모: {profile["notes"] or "없음"}

현재 냉장고 식재료:
{", ".join(ingredients) if ingredients else "없음"}

사용자 요청:
{user_request}

규칙:
1) 냉장고 식재료를 최우선으로 사용
2) 단순당 과다, 과도한 정제 탄수화물, 튀김류는 피하기
3) 단백질, 식이섬유, 채소 우선
4) 한국어로 간단명료하게
5) 아래 형식으로 답하기

형식:
[추천 식단]
- 메뉴명:
- 왜 괜찮은지:
- 사용하는 재료:
- 간단 조리법(3단계 이내):
- 주의할 점:

[대체 메뉴 1개]
- 메뉴명:
- 사용하는 재료:
- 한 줄 설명:
""".strip()


def request_meal_plan_from_openai(user_request: str) -> str:
    client = get_openai_client()
    if client is None:
        return "OpenAI API 키가 설정되지 않았습니다."

    if not st.session_state.ingredients:
        return "먼저 냉장고 식재료를 저장해 주세요."

    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "당신은 당뇨 및 다이어트 식단 보조 코치입니다. 의료행위는 하지 않습니다.",
                },
                {
                    "role": "user",
                    "content": build_meal_plan_prompt(user_request),
                },
            ],
        )

        text = getattr(response, "output_text", None)
        if text and text.strip():
            return text.strip()
        return "식단 추천 생성에 실패했습니다."
    except Exception as e:
        return f"식단 추천 생성 중 오류가 발생했습니다: {e}"


# =========================
# 메시지 처리
# =========================
def append_chat(user_input: str, bot_output: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": bot_output})


def submit_message(user_input: str) -> None:
    if is_meal_plan_request(user_input):
        response = request_meal_plan_from_openai(user_input)
        entry_type = "meal_plan"
        blood_sugar = None
    else:
        response, blood_sugar, entry_type = generate_basic_response(user_input)

    append_chat(user_input, response)
    insert_log_to_db(entry_type, blood_sugar, user_input, response)
    load_logs_from_db()


# =========================
# 앱 초기 DB 동기화
# =========================
def ensure_loaded_once() -> None:
    if st.session_state.initialized:
        return

    load_profile_from_db()
    load_logs_from_db()
    st.session_state.initialized = True


# =========================
# UI
# =========================
def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    ensure_loaded_once()

    with st.sidebar:
        st.header("설정")

        st.text_input(
            "사용자 ID",
            key="user_id",
            help="가족용 단일 계정이면 dad_001처럼 고정해서 쓰면 됩니다.",
        )

        if st.button("DB에서 다시 불러오기", use_container_width=True):
            load_profile_from_db()
            load_logs_from_db()
            st.success("다시 불러왔습니다.")

        st.subheader("건강 목표")
        st.session_state.profile["goal"] = st.selectbox(
            "목표",
            [
                "당뇨 관리 + 체중 감량",
                "당뇨 관리 중심",
                "체중 감량 중심",
                "혈당 안정 + 포만감 유지",
            ],
            index=[
                "당뇨 관리 + 체중 감량",
                "당뇨 관리 중심",
                "체중 감량 중심",
                "혈당 안정 + 포만감 유지",
            ].index(
                st.session_state.profile.get("goal", "당뇨 관리 + 체중 감량")
            )
            if st.session_state.profile.get("goal") in [
                "당뇨 관리 + 체중 감량",
                "당뇨 관리 중심",
                "체중 감량 중심",
                "혈당 안정 + 포만감 유지",
            ]
            else 0,
        )

        st.session_state.profile["avoid_foods"] = st.text_input(
            "피하고 싶은 음식",
            value=st.session_state.profile["avoid_foods"],
            placeholder="예: 빵, 라면, 단 음료",
        )

        st.session_state.profile["notes"] = st.text_area(
            "추가 메모",
            value=st.session_state.profile["notes"],
            placeholder="예: 저녁은 가볍게, 씹기 쉬운 음식 선호",
            height=80,
        )

        st.subheader("냉장고 식재료")
        ingredient_input = st.text_area(
            "쉼표(,)나 줄바꿈으로 입력",
            value=", ".join(st.session_state.ingredients),
            placeholder="예: 닭가슴살, 두부, 계란, 오이, 양배추, 브로콜리, 현미밥",
            height=120,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("식재료 저장", use_container_width=True):
                st.session_state.ingredients = normalize_ingredients(ingredient_input)
                ok, msg = save_profile_to_db()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        with c2:
            if st.button("프로필 전체 저장", use_container_width=True):
                st.session_state.ingredients = normalize_ingredients(ingredient_input)
                ok, msg = save_profile_to_db()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.caption(f"현재 저장 대상 재료: {ingredients_to_text()}")

    st.title(APP_TITLE)
    st.caption("외부 DB에 저장되어 웹을 껐다 켜도 기억되는 버전")
    st.info(SYSTEM_GUIDE)
    st.warning(DISCLAIMER)

    st.subheader("빠른 요청")
    a, b, c = st.columns(3)

    with a:
        if st.button("아침 식단 추천", use_container_width=True):
            submit_message("아침 식단 추천해줘")
            st.rerun()

    with b:
        if st.button("점심 식단 추천", use_container_width=True):
            submit_message("점심 식단 추천해줘")
            st.rerun()

    with c:
        if st.button("저녁 식단 추천", use_container_width=True):
            submit_message("저녁 식단 추천해줘")
            st.rerun()

    st.divider()

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.chat_input("예: 혈당 130 / 점심 먹었어 / 저녁 식단 추천해줘")
    if user_input:
        submit_message(user_input)
        st.rerun()

    st.divider()

    st.subheader("최근 기록")
    if st.session_state.logs_df.empty:
        st.write("아직 저장된 기록이 없습니다.")
    else:
        st.dataframe(st.session_state.logs_df, use_container_width=True)

    csv_bytes = st.session_state.logs_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="기록 CSV 다운로드",
        data=csv_bytes,
        file_name=f"diabetes_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()