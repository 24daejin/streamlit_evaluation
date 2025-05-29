import streamlit as st
import json
import uuid
from datetime import datetime
import os
import pandas as pd
from openai import OpenAI
import traceback  # 오류 추적용

# 페이지 기본 설정
st.set_page_config(
    page_title="기후 위기 스토리보드 작성 활동 도우미",
    page_icon="🌍",
    layout="wide"
)

# API 키 설정 - 환경 변수나 직접 입력으로 수정
# 환경 변수 사용
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# 2. 환경 변수에 없으면 Streamlit Secrets에서 가져오기
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets에서 'OPENAI_API_KEY'를 설정해주세요.")
        st.stop()

# 3. API 키 유효성 검사
if not OPENAI_API_KEY:
    st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets에서 'OPENAI_API_KEY'를 설정해주세요.")
    st.stop()
    
# 학생별 최대 API 호출 횟수 설정
MAX_API_CALLS_PER_STUDENT = 50  # 학생별 최대 API 호출 횟수

# 학생별 API 호출 횟수 추적
if "student_api_calls" not in st.session_state:
    st.session_state.student_api_calls = {}

# GPT API 호출 함수 (모델 선택 가능)
def get_gpt_response(messages, use_gpt4=False):
    student_id = st.session_state.student_id
    
    # 학생별 API 호출 횟수 초기화
    if student_id not in st.session_state.student_api_calls:
        st.session_state.student_api_calls[student_id] = 0
    
    # API 호출 횟수 제한 확인
    if st.session_state.student_api_calls[student_id] >= MAX_API_CALLS_PER_STUDENT:
        return "API 호출 횟수가 제한에 도달했습니다. 선생님에게 문의해주세요."
    
    # 간단한 캐싱을 위한 키 생성 (사용자 메시지만 고려)
    cache_key = str([msg for msg in messages if msg["role"] == "user"]) + str(use_gpt4)
    
    # 캐시된 응답이 있는지 확인
    if cache_key in st.session_state.response_cache:
        return st.session_state.response_cache[cache_key]
    
    try:
        # API 호출 카운터 증가
        st.session_state.api_call_count += 1
        st.session_state.student_api_calls[student_id] += 1
        
        # 모델 선택
        model = FEEDBACK_MODEL if use_gpt4 else DEFAULT_MODEL
        
        api_params = {
            "model": model,
            "messages": messages
        }
        # o1 모델이 아닌 경우에만 temperature 추가
        if not model.startswith("o1"):
            api_params["temperature"] = 0.7
            response = client.chat.completions.create(**api_params)
        
        response_text = response.choices[0].message.content
        
        # 응답 캐싱
        st.session_state.response_cache[cache_key] = response_text
        
        return response_text
    except Exception as e:
        st.error(f"GPT 응답 생성 중 오류가 발생했습니다: {str(e)}")
        return "죄송합니다, 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해 주세요."
# GPT를 활용한 메시지 관련성 분석 함수들 (get_gpt_response 함수 아래에 추가)

def analyze_message_relevance(message_content):
    """GPT를 사용해서 메시지가 스토리보드 관련인지 판단"""
    
    analysis_prompt = f"""
    다음 학생의 메시지가 기후 위기 스토리보드 작성과 관련된 의미있는 내용인지 판단해주세요.
    
    학생 메시지: "{message_content}"
    
    다음 기준으로 판단해주세요:
    
    관련된 내용:
    - 스토리보드 주제, 구성, 캐릭터, 장면에 대한 질문이나 아이디어
    - 기후 위기 관련 내용 문의 및 토론
    - 창작 과정에서의 구체적인 고민이나 요청
    - 스토리보드 제작 방법에 대한 질문
    - 발표 준비나 피드백 요청
    - 구체적인 시나리오나 상황 설정에 대한 논의
    
    관련없는 내용:
    - 단순 인사말 ("안녕하세요", "감사합니다", "네", "좋아요")
    - 수행평가와 무관한 잡담이나 개인적인 이야기
    - 너무 짧거나 의미없는 응답 ("몰라요", "음", "어")
    - 단순 확인 응답 ("알겠습니다", "네 맞아요")
    
    답변: "관련됨" 또는 "관련없음" 중 하나만 정확히 답하세요.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.1,  # 일관성을 위해 낮은 temperature
        )
        
        result = response.choices[0].message.content.strip()
        return "관련됨" in result
    except Exception as e:
        # API 오류 시 보수적으로 관련됨으로 처리 (학생에게 불이익 방지)
        print(f"메시지 분석 중 오류: {str(e)}")
        return True

def count_relevant_prompts(conversation):
    """대화에서 스토리보드 관련 프롬프트 수 계산"""
    relevant_count = 0
    
    for msg in conversation["messages"]:
        if msg["role"] == "user":
            # 시스템 메시지나 웰컴 메시지에 대한 단순 응답은 제외
            if len(msg["content"].strip()) < 3:
                continue
                
            if analyze_message_relevance(msg["content"]):
                relevant_count += 1
    
    return relevant_count

def analyze_conversations_with_gpt(conversations, progress_callback=None):
    """여러 대화를 GPT로 분석 (진행 상황 표시 포함)"""
    analyzed_data = []
    total = len(conversations)
    
    for i, conv in enumerate(conversations):
        if progress_callback:
            progress_callback(i + 1, total)
            
        student_name = conv["student_name"]
        student_id = conv["student_id"]
        
        # GPT로 관련 메시지 수 분석
        relevant_count = count_relevant_prompts(conv)
        total_user_messages = sum(1 for msg in conv["messages"] if msg["role"] == "user")
        assistant_msg_count = sum(1 for msg in conv["messages"] if msg["role"] == "assistant")
        
        # 대화 시간 계산
        if conv["messages"]:
            first_msg_time = datetime.strptime(conv["messages"][0]["timestamp"], "%Y-%m-%d %H:%M:%S")
            last_msg_time = datetime.strptime(conv["messages"][-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
            duration = (last_msg_time - first_msg_time).total_seconds() / 60
        else:
            duration = 0
        
        # 피드백 여부 확인
        has_feedback = "feedback" in conv and len(conv["feedback"]) > 0
        
        analyzed_data.append({
            "학생명": student_name,
            "학번": student_id,
            "관련 프롬프트 수": relevant_count,
            "전체 메시지 수": total_user_messages,
            "AI 응답 수": assistant_msg_count,
            "관련도": f"{relevant_count}/{total_user_messages}" if total_user_messages > 0 else "0/0",
            "대화 시간(분)": round(duration, 1),
            "피드백 여부": "O" if has_feedback else "X"
        })
    
    return analyzed_data
    
# 데이터 저장 경로 설정
DATA_DIR = "data"
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# 학생 정보 파일 초기화
if not os.path.exists(STUDENTS_FILE):
    with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

# OpenAI 클라이언트 초기화 (한 번만 생성)
client = OpenAI(api_key=OPENAI_API_KEY)

# 모델 설정 (GPT-04-mini 또는 GPT-4o)
DEFAULT_MODEL = "gpt-o4-mini"  # 기본 모델
FEEDBACK_MODEL = "gpt-4o"  # 피드백에 사용할 모델

# 세션 ID 생성 (처음 앱 실행 시 한 번만)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 메시지 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 시스템 메시지 추가
    st.session_state.messages.append({
        "role": "system",
        "content": """학생들의 수행평가를 도움을 주기위한 대화를 하려고 하는데, 역할, 말투, 핵심 주제, 수행평가 단계, 제한사항, 참고사항을 고려하여 응답해주세요. 
        # 역할 : 중학교 3학년 학생들이 기후 위기 관련 스토리보드를 작성하는 것을 돕는 조수
        # 말투 : 학생들에게 친절하고 이해하기 쉬운 언어로 응답
        # 핵심 주제
        ## 기후위기
        ### 소비는 탄소 발자국을 남긴다
        - **스마트폰과 자원 소비**: 스마트폰 생산에 40여 가지 광물이 사용되며, 평균 교체 주기는 2.7년
        - **데이터 센터의 환경 영향**: 전 세계 이산화탄소 배출의 2%가 데이터 센터에서 발생
        - **플라스틱 문제**: 1950년 200만톤 생산에서 2015년 4억 7000만톤으로 증가
        - **패스트 패션의 영향**: 2000년 500억벌에서 2015년 1000억벌로 판매량 증가

        ### 우리가 먹는 것 하나하나가
        - **고기 소비와 환경**: 축산업은 직접 이산화탄소 배출의 18%, 간접 포함 시 30% 차지
        - **초콜릿과 카카오 재배**: 지난 50년간 코트디부아르 숲의 80%가 사라짐
        - **새우 양식과 맹그로브 숲**: 맹그로브 숲은 탄소 흡수력이 열대우림의 2.5배이나 새우 양식으로 파괴됨
        - **음식물 쓰레기**: 생산된 음식의 1/3은 먹기도 전에 버려짐

        ### 남극이 펭귄을 잃게 될 때
        - **북극 빙하**: 30년간 북극 빙하 50%가 감소, 2035년에는 해빙이 없을 것으로 예상
        - **영구동토층 융해**: 메탄 발생과 감염병 확산 위험
        - **남극 기온 상승**: 최근 50년간 3도 상승하여 펭귄 서식에 위협
        - **물 순환 문제**: 가뭄과 폭우의 반복으로 수자원 위기

        ### 기후위기에 대응하는 우리의 실천
        - **화석연료 기업의 영향**: 최근 50년간 전 세계 온실가스 배출량의 35% 차지
        - **친환경 교통**: 자전거 친화 도시의 확산과 공유 차량 시스템
        - **재생에너지 확대**: 화석연료 중심에서 재생에너지 중심 전환 필요
        - **지속가능한 생활방식**: 라벨 없는 상품, 텀블러 공유 서비스 등 새로운 시도
        
        # 수행평가 단계
        1. 모둠별 활동(스토리보드 주제 선정)
        * 넓은 주제의 주제보다는 좁은 범위의 주제 선정(예: 데이터 센터 → 데이터 센터의 위치, 맹그로브 숲 → 맹그로브 숲의 영향)
        * 주제를 잘 표현하기 위해서는 몇 장의 스토리보드가 적절한지 결정
        * 모둠원들이 공유해야 하는 스토리보드의 전체 분위기 및 등장인물, 배경등 미리 정하기
        2. 개인별 스토리보드 작성
        * 모둠의 스토리보드 중에서 역할 분담받은 특정 장면의 스토리보드를 만들기
        * 특정 장면을 표현하기 위한 몇 가지 컷 만들기
        * 각 컷의 비디오(이미지)와 해당 컷의 설명, 대략적인 소요시간 표현하기
        3. 발표
        * 모둠에서 작성한 스토리보드를 한 명이 발표
        * 발표시에 어떠한 주제를 어떠한 분위기에서 어떤 인물이 어떻게 했다는 것을 표현
        # 제한사항
        * 수행평가임을 분명히 하고 대화에서 수행평가와 관련없는 대화가 실시될 경우에는 감점이 될 수 있음.
        * 수행평가 관련 대화가 아닌 대화를 3번 연속해서 진행할 시에 경고하기.
        # 참고사항
        1. 모둠 구성 : 3 ~ 4명
        2. 학생들은 수행평가 단계1, 단계2, 단계3 에서 도움 요청 예정
        3. 요청하는 단계를 이해한후 대답 요구
        4. 창의적이고 효과적인 스토리보드 제작하기 위해 도움을 줌"""
    })

# 학생 정보 상태 초기화
if "student_info_submitted" not in st.session_state:
    st.session_state.student_info_submitted = False

# 피드백 모드 상태 초기화
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False

# API 사용량 모니터링
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0

# 캐싱 설정
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}


# 학생 정보 저장
def save_student_info(data):
    try:
        # 기존 학생 정보 불러오기
        with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
            students = json.load(f)

        # 새 학생 정보 추가
        students.append(data)

        # 업데이트된 정보 저장
        with open(STUDENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(students, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"학생 정보 저장 중 오류 발생: {str(e)}")
        return False


# 대화 저장 - 파일명 형식 변경: 학번_이름.json
def save_conversation(data):
    student_id = data["student_id"]
    student_name = data["student_name"]
    # 파일명 형식 변경
    filename = f"{student_id}_{student_name}.json"
    conversation_file = os.path.join(CONVERSATIONS_DIR, filename)

    try:
        # 기존 대화 불러오기 또는 초기화
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
        else:
            conversation = {
                "session_id": data["session_id"],
                "student_name": student_name,
                "student_id": student_id,
                "messages": []
            }

        # 새 메시지 추가
        message = {
            "role": data["type"].split("_")[0],  # "user" 또는 "assistant"
            "content": data["content"],
            "timestamp": data["timestamp"]
        }
        conversation["messages"].append(message)

        # 업데이트된 대화 저장
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"대화 저장 중 오류 발생: {str(e)}")
        return False


# 피드백 저장 - 파일명 형식 변경: 학번_이름.json
def save_feedback(data):
    student_id = data["student_id"]
    student_name = data["student_name"]
    # 파일명 형식 변경
    filename = f"{student_id}_{student_name}.json"
    conversation_file = os.path.join(CONVERSATIONS_DIR, filename)

    try:
        # 기존 대화 불러오기
        if os.path.exists(conversation_file):
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)

            # 피드백 추가
            if "feedback" not in conversation:
                conversation["feedback"] = []

            feedback = {
                "content": data["content"],
                "timestamp": data["timestamp"]
            }
            conversation["feedback"].append(feedback)

            # 업데이트된 대화 저장
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)

            return True
        else:
            st.error(f"대화 파일을 찾을 수 없습니다: {conversation_file}")
            return False
    except Exception as e:
        st.error(f"피드백 저장 중 오류 발생: {str(e)}")
        return False


# 데이터 저장 함수
def save_data(data):
    if data["type"] == "student_info":
        return save_student_info(data)
    elif data["type"] == "feedback":
        return save_feedback(data)
    else:  # user_message 또는 assistant_message
        return save_conversation(data)


# GPT API 호출 함수 (모델 선택 가능)
def get_gpt_response(messages, use_gpt4=False):
    # 간단한 캐싱을 위한 키 생성 (사용자 메시지만 고려)
    cache_key = str([msg for msg in messages if msg["role"] == "user"]) + str(use_gpt4)

    # 캐시된 응답이 있는지 확인
    if cache_key in st.session_state.response_cache:
        return st.session_state.response_cache[cache_key]

    try:
        # API 호출 카운터 증가
        st.session_state.api_call_count += 1

        # 모델 선택
        model = FEEDBACK_MODEL if use_gpt4 else DEFAULT_MODEL

        # API 호출
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )

        response_text = response.choices[0].message.content

        # 응답 캐싱
        st.session_state.response_cache[cache_key] = response_text

        return response_text
    except Exception as e:
        st.error(f"GPT 응답 생성 중 오류가 발생했습니다: {str(e)}")
        return "죄송합니다, 응답을 생성하는 중에 오류가 발생했습니다. 다시 시도해 주세요."


# 사이드바 - 스토리보드 작성 가이드
with st.sidebar:
    st.title("스토리보드 작성 가이드")
    st.markdown("""
    ### 스토리보드란?
    스토리보드는 이야기의 흐름을 시각적으로 계획하는 도구입니다. 기후 위기에 관한 
    여러분의 생각과 아이디어를 시각화하는 데 도움이 됩니다.

    ### 효과적인 프롬프트 작성법
    1. **구체적인 상황 설정하기**: "피자 가게에서 새우가 토핑으로 올라간 피자를 먹으면서 친구들과 이야기 중인 상황을 그린다면?"
    2. **인물과 감정 추가하기**: "새우를 먹다가 맹그로브 숲이 사라져 간다는 것을 알게된 후 피자를 먹을 때 느끼는 감정은?"
    3. **문제 해결 방식 탐색하기**: "맹그로브 숲이 사라지는 것을 막기 위해서는 뭘 해야할까?"
    4. **대비 활용하기**: "현재와 맹그로브 숲이 사라진 미래의 환경을 대비하여 보여준다면?"
    5. **지역 특성 반영하기**: "우리 지역에서 볼 수 있는 기후 변화의 신호는?"

    ### 평가 기준
    #### 스토리보드 평가 (40점)
    - **A등급 (40점)**: 5개 이상의 효과적인 프롬프트를 작성하면서 기존의 문제점을 정확하게 파악하고 체계적으로 개선함
    - **B등급 (35점)**: 4개의 효과적인 프롬프트를 작성하면서 기존의 문제점을 정확하게 파악하고 체계적으로 개선함
    - **C등급 (30점)**: 3개의 효과적인 프롬프트를 작성하면서 문제점 파악과 개선이 대체적으로 체계적
    - **D등급 (25점)**: 2개의 기본적인 프롬프트를 작성
    - **E등급 (20점)**: 1개의 단순한 프롬프트만 사용

    #### 발표 평가 (20점)
    - **A등급 (20점)**: 핵심 메시지를 기후 위기와 관련지어 명확하게 발표
    - **B등급 (15점)**: 핵심 메시지를 기후 위기와 관련지었지만 명확하게 전달되지 않음
    - **C등급 (10점)**: 핵심 메시지를 기후 위기와 관련짓지 않고 발표
    """)

    # 피드백 받기 버튼
    if st.button("내 스토리보드 피드백 받기"):
        if len([m for m in st.session_state.messages if m["role"] == "user"]) > 0:
            st.session_state.feedback_mode = True
            st.rerun()
        else:
            st.warning("먼저 스토리보드 작성을 위한 대화가 필요합니다.")

    # 책 내용 요약 확장 섹션
    with st.expander("📚 필독서 내용 요약"):
        st.markdown("""
        ### 소비는 탄소 발자국을 남긴다
        - **스마트폰과 자원 소비**: 스마트폰 생산에 40여 가지 광물이 사용되며, 평균 교체 주기는 2.7년
        - **데이터 센터의 환경 영향**: 전 세계 이산화탄소 배출의 2%가 데이터 센터에서 발생
        - **플라스틱 문제**: 1950년 200만톤 생산에서 2015년 4억 7000만톤으로 증가
        - **패스트 패션의 영향**: 2000년 500억벌에서 2015년 1000억벌로 판매량 증가

        ### 우리가 먹는 것 하나하나가
        - **고기 소비와 환경**: 축산업은 직접 이산화탄소 배출의 18%, 간접 포함 시 30% 차지
        - **초콜릿과 카카오 재배**: 지난 50년간 코트디부아르 숲의 80%가 사라짐
        - **새우 양식과 맹그로브 숲**: 맹그로브 숲은 탄소 흡수력이 열대우림의 2.5배이나 새우 양식으로 파괴됨
        - **음식물 쓰레기**: 생산된 음식의 1/3은 먹기도 전에 버려짐

        ### 남극이 펭귄을 잃게 될 때
        - **북극 빙하**: 30년간 북극 빙하 50%가 감소, 2035년에는 해빙이 없을 것으로 예상
        - **영구동토층 융해**: 메탄 발생과 감염병 확산 위험
        - **남극 기온 상승**: 최근 50년간 3도 상승하여 펭귄 서식에 위협
        - **물 순환 문제**: 가뭄과 폭우의 반복으로 수자원 위기

        ### 기후위기에 대응하는 우리의 실천
        - **화석연료 기업의 영향**: 최근 50년간 전 세계 온실가스 배출량의 35% 차지
        - **친환경 교통**: 자전거 친화 도시의 확산과 공유 차량 시스템
        - **재생에너지 확대**: 화석연료 중심에서 재생에너지 중심 전환 필요
        - **지속가능한 생활방식**: 라벨 없는 상품, 텀블러 공유 서비스 등 새로운 시도
        """)

# 메인 화면
st.title("🌍 기후 위기 스토리보드 작성 활동 도우미")

# 학생 정보 입력 폼
if not st.session_state.student_info_submitted:
    with st.form("student_info_form"):
        st.subheader("학생 정보 입력")
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("이름")
        with col2:
            student_id = st.text_input("학번")

        submitted = st.form_submit_button("로그인")
        if submitted and student_name and student_id:
            st.session_state.student_name = student_name
            st.session_state.student_id = student_id
            st.session_state.student_info_submitted = True

            # 학생 정보 저장
            student_info = {
                "session_id": st.session_state.session_id,
                "student_name": student_name,
                "student_id": student_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "student_info"
            }

            # 데이터 저장
            save_data(student_info)

            # 웰컴 메시지 추가
            welcome_message = {
                "role": "assistant",
                "content": f"""안녕하세요, {student_name} 학생! 기후 위기 스토리보드 작성을 도와드릴게요.
저는 스토리보드 모둠활동, 개별활동, 발표준비에 도움을 드릴 수 있어요.
그리고 여러분이 만들 스토리보드는 기후 위기에 관한 중요한 메시지를 전달하는 도구가 될 거예요. 
어떤 아이디어나 질문이 있으신가요?

예를 들어:
- 어떤 주제로 하는 것이 좋을까?
- 어떤 형식으로 스토리보드를 만들고 싶으신가요? (짧은 만화, 시나리오, 광고 등)
- 스토리보드에 포함되어야 하는 내용은 무엇일까?
- 스토리보드에 어떤 캐릭터나 상황을 포함시키고 싶으신가요?
- 발표할 때에는 어떤 점을 위주로 발표하면 좋을까?

자유롭게 질문하거나 아이디어를 나눠주세요!
"""
            }
            st.session_state.messages.append(welcome_message)

            # 웰컴 메시지도 저장
            welcome_data = {
                "session_id": st.session_state.session_id,
                "student_name": student_name,
                "student_id": student_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "assistant_message",
                "content": welcome_message["content"]
            }
            save_data(welcome_data)

            st.rerun()

    # 학생 정보가 입력되지 않은 경우 추가 내용 표시하지 않음
    st.info("위의 학생 정보를 입력하신 후 스토리보드 작성을 시작할 수 있습니다.")

# 학생 정보가 제출된 경우에만 챗 인터페이스 표시
elif st.session_state.student_info_submitted:
    # 피드백 모드인 경우
    if st.session_state.feedback_mode:
        st.subheader("스토리보드 피드백")

        # 지금까지의 대화 내용을 바탕으로 피드백 요청
        with st.spinner("피드백을 생성 중입니다..."):
            # 현재까지의 대화 내용을 바탕으로 프롬프트 분석 및 피드백
            feedback_prompt = """지금까지의 대화를 바탕으로 내 스토리보드 작업에 대해 다음 항목에 대한 피드백을 제공해주세요:
            1. 수행평가와 관련되어 사용한 프롬프트의 수와 질 (평가 기준에 따른 현재 등급)
            2. 스토리보드의 기후 위기 관련성
            3. 개선할 점과 강화할 점
            4. 발표 시 핵심적으로 강조해야 할 메시지

            피드백은 구체적이고 건설적이며 격려하는 방식으로 제공해주세요."""

            # 피드백 생성을 위한 메시지 구성
            feedback_messages = [m for m in st.session_state.messages]
            feedback_messages.append({"role": "user", "content": feedback_prompt})

            # GPT-4를 사용하여 피드백 생성 (더 높은 품질의 피드백을 위해)
            feedback = get_gpt_response(feedback_messages, use_gpt4=True)

            # 피드백 표시
            st.markdown(f"### 피드백 결과\n{feedback}")

            # 피드백 기록
            feedback_data = {
                "session_id": st.session_state.session_id,
                "student_name": st.session_state.student_name,
                "student_id": st.session_state.student_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "feedback",
                "content": feedback
            }

            # 데이터 저장
            save_data(feedback_data)

        # 피드백 모드 종료 버튼
        if st.button("스토리보드 작성으로 돌아가기"):
            st.session_state.feedback_mode = False
            st.rerun()

    # 일반 채팅 모드
    else:
        # 메시지 기록 표시
        for msg in st.session_state.messages:
            if msg["role"] != "system":  # 시스템 메시지는 표시하지 않음
                if msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
                elif msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])

        # 사용자 입력 받기
        user_input = st.chat_input("스토리보드에 대해 질문하거나 아이디어를 입력하세요...")

        if user_input:
            # 사용자 메시지 추가 및 표시
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 입력 시간 기록
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 대화 로그 저장
            chat_log = {
                "session_id": st.session_state.session_id,
                "student_name": st.session_state.student_name,
                "student_id": st.session_state.student_id,
                "timestamp": current_time,
                "type": "user_message",
                "content": user_input
            }

            # 데이터 저장
            save_data(chat_log)

            # GPT 응답 생성 (기본 모델 사용)
            with st.spinner("응답을 생성 중입니다..."):
                messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                response = get_gpt_response(messages_for_api, use_gpt4=False)

            # 응답 메시지 추가 및 표시
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            # 응답 로그 저장
            response_log = {
                "session_id": st.session_state.session_id,
                "student_name": st.session_state.student_name,
                "student_id": st.session_state.student_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "assistant_message",
                "content": response
            }

            # 데이터 저장
            save_data(response_log)

# 관리자 모드 (숨겨진 기능) - URL에 ?admin=true 추가 시 접근 가능
if st.query_params.get("admin", "false") == "true":
    st.markdown("---")
    st.header("👨‍🏫 관리자 대시보드")

    # API 사용 통계
    st.metric("총 API 호출 횟수", st.session_state.api_call_count)

    # 탭 생성
    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs(["학생 목록", "대화 내용", "데이터 분석", "백업 다운로드"])

    with admin_tab1:
        st.subheader("등록된 학생 목록")

        # 학생 정보 불러오기
        try:
            with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
                students = json.load(f)

            if students:
                # 학생 정보를 DataFrame으로 변환
                student_df = pd.DataFrame(students)
                st.dataframe(student_df)

                # 학생 수 표시
                st.info(f"총 {len(students)}명의 학생이 등록되었습니다.")

                # 학생 정보 다운로드 버튼
                csv = student_df.to_csv(index=False)
                st.download_button(
                    label="학생 목록 다운로드 (CSV)",
                    data=csv,
                    file_name="학생목록.csv",
                    mime="text/csv"
                )
            else:
                st.info("아직 등록된 학생이 없습니다.")
        except Exception as e:
            st.error(f"학생 정보 로드 중 오류: {str(e)}")

    with admin_tab2:
        st.subheader("학생별 대화 내용")

        # 학생 선택 드롭다운
        try:
            with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
                students = json.load(f)

            if students:
                student_options = [f"{s['student_name']} ({s['student_id']})" for s in students]
                selected_student = st.selectbox("학생 선택", options=student_options)

                # 선택한 학생의 정보 찾기
                selected_name, selected_id = selected_student.split(" (")
                selected_id = selected_id.rstrip(")")

                # 대화 파일 찾기 (파일명 형식: 학번_이름.json)
                conversation_file = os.path.join(CONVERSATIONS_DIR, f"{selected_id}_{selected_name}.json")

                if os.path.exists(conversation_file):
                    with open(conversation_file, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)

                    # 대화 내용 표시
                    for msg in conversation["messages"]:
                        if msg["role"] == "user":
                            st.info(f"**학생 ({msg['timestamp']}):**\n{msg['content']}")
                        elif msg["role"] == "assistant":
                            st.success(f"**AI ({msg['timestamp']}):**\n{msg['content']}")

                    # 피드백 표시
                    if "feedback" in conversation and conversation["feedback"]:
                        st.subheader("피드백 기록")
                        for feedback in conversation["feedback"]:
                            st.warning(f"**피드백 ({feedback['timestamp']}):**\n{feedback['content']}")

                    # 다운로드 버튼
                    conversation_json = json.dumps(conversation, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="대화 내용 다운로드 (JSON)",
                        data=conversation_json,
                        file_name=f"{selected_id}_{selected_name}_대화.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"대화 파일을 찾을 수 없습니다: {conversation_file}")
            else:
                st.info("아직 등록된 학생이 없습니다.")
        except Exception as e:
            st.error(f"대화 내용 로드 중 오류: {str(e)}")

    with admin_tab3:
        st.subheader("데이터 분석")

        # 전체 데이터 분석
        try:
            # 모든 대화 파일 불러오기
            all_conversations = []
            for filename in os.listdir(CONVERSATIONS_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(CONVERSATIONS_DIR, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        all_conversations.append(conversation)
            
            if all_conversations:
                # 분석 방법 선택
                analysis_method = st.radio(
                    "분석 방법 선택:",
                    ["빠른 분석 (기존 방식)", "정밀 분석 (GPT 활용)"],
                    help="정밀 분석은 GPT를 사용하여 더 정확하지만 시간이 더 걸립니다."
                )
                
                if analysis_method == "정밀 분석 (GPT 활용)":
                    # GPT 분석 실행 버튼
                    if st.button("GPT 분석 시작", type="primary"):
                        # 진행 상황 표시
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(current, total):
                            progress = current / total
                            progress_bar.progress(progress)
                            status_text.text(f'분석 진행 중... {current}/{total} ({progress:.1%})')
                        
                        # GPT 분석 실행
                        with st.spinner("GPT를 활용한 정밀 분석 중..."):
                            student_message_data = analyze_conversations_with_gpt(
                                all_conversations, 
                                progress_callback=update_progress
                            )
                        
                        # 진행 상황 정리
                        progress_bar.empty()
                        status_text.text("분석 완료!")
                        
                        # 세션에 결과 저장
                        st.session_state.gpt_analysis_result = student_message_data
                    
                    # 저장된 분석 결과가 있으면 표시
                    if hasattr(st.session_state, 'gpt_analysis_result'):
                        student_message_data = st.session_state.gpt_analysis_result
                        
                        # 프롬프트 수에 따른 등급 평가 함수
                        def evaluate_grade(prompt_count):
                            if prompt_count >= 5:
                                return "A (40점)"
                            elif prompt_count == 4:
                                return "B (35점)"  
                            elif prompt_count == 3:
                                return "C (30점)"
                            elif prompt_count == 2:
                                return "D (25점)"
                            else:
                                return "E (20점)"
                        
                        # 데이터프레임 생성 및 등급 추가
                        student_df = pd.DataFrame(student_message_data)
                        student_df["예상 등급"] = student_df["관련 프롬프트 수"].apply(evaluate_grade)
                        
                        # 기본 통계 표시
                        st.subheader("GPT 분석 결과")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_relevant = student_df["관련 프롬프트 수"].mean()
                            st.metric("평균 관련 프롬프트 수", f"{avg_relevant:.1f}")
                        with col2:
                            avg_total = student_df["전체 메시지 수"].mean()
                            st.metric("평균 전체 메시지 수", f"{avg_total:.1f}")
                        with col3:
                            relevance_rate = (student_df["관련 프롬프트 수"].sum() / student_df["전체 메시지 수"].sum()) * 100
                            st.metric("전체 관련도", f"{relevance_rate:.1f}%")
                        
                        # 상세 분석 테이블
                        st.subheader("학생별 상세 분석")
                        st.dataframe(
                            student_df.sort_values(by="관련 프롬프트 수", ascending=False),
                            use_container_width=True
                        )
                        
                        # 등급 분포 차트
                        st.subheader("등급 분포 (GPT 분석 기준)")
                        grade_counts = student_df["예상 등급"].value_counts().reset_index()
                        grade_counts.columns = ["등급", "학생 수"]
                        st.bar_chart(grade_counts.set_index("등급"))
                        
                        # 관련도 분석 차트
                        st.subheader("학생별 프롬프트 관련도")
                        chart_data = student_df[["학생명", "관련 프롬프트 수", "전체 메시지 수"]].head(10)
                        st.bar_chart(chart_data.set_index("학생명"))
                        
                        # GPT 분석 결과 다운로드
                        csv_gpt = student_df.to_csv(index=False)
                        st.download_button(
                            label="GPT 분석 결과 다운로드 (CSV)",
                            data=csv_gpt,
                            file_name="GPT_분석_결과.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.info("👆 위의 'GPT 분석 시작' 버튼을 클릭하여 정밀 분석을 시작하세요.")
                
                else:
                    # 기존 빠른 분석 방식
                    total_messages = sum(len(conv["messages"]) for conv in all_conversations)
                    user_messages = sum(
                        sum(1 for msg in conv["messages"] if msg["role"] == "user") for conv in all_conversations)
                    assistant_messages = sum(
                        sum(1 for msg in conv["messages"] if msg["role"] == "assistant") for conv in all_conversations)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 메시지 수", total_messages)
                    with col2:
                        st.metric("학생 메시지 수", user_messages)
                    with col3:
                        st.metric("AI 응답 수", assistant_messages)

                    # 기존 분석 로직 (간단한 메시지 수 기반)
                    student_message_data = []
                    for conv in all_conversations:
                        student_name = conv["student_name"]
                        student_id = conv["student_id"]
                        user_msg_count = sum(1 for msg in conv["messages"] if msg["role"] == "user")
                        assistant_msg_count = sum(1 for msg in conv["messages"] if msg["role"] == "assistant")

                        if conv["messages"]:
                            first_msg_time = datetime.strptime(conv["messages"][0]["timestamp"], "%Y-%m-%d %H:%M:%S")
                            last_msg_time = datetime.strptime(conv["messages"][-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
                            duration = (last_msg_time - first_msg_time).total_seconds() / 60
                        else:
                            duration = 0

                        has_feedback = "feedback" in conv and len(conv["feedback"]) > 0

                        student_message_data.append({
                            "학생명": student_name,
                            "학번": student_id,
                            "학생 메시지 수": user_msg_count,
                            "AI 응답 수": assistant_msg_count,
                            "대화 시간(분)": round(duration, 1),
                            "피드백 여부": "O" if has_feedback else "X"
                        })

                    # 기존 평가 함수
                    def evaluate_grade_simple(prompt_count):
                        if prompt_count >= 5:
                            return "A (40점)"
                        elif prompt_count == 4:
                            return "B (35점)"
                        elif prompt_count == 3:
                            return "C (30점)"
                        elif prompt_count == 2:
                            return "D (25점)"
                        else:
                            return "E (20점)"

                    student_df = pd.DataFrame(student_message_data)
                    student_df["예상 등급"] = student_df["학생 메시지 수"].apply(evaluate_grade_simple)

                    st.subheader("학생별 기본 분석 (메시지 수 기준)")
                    st.dataframe(student_df)
                    
                    # 기존 차트들
                    st.subheader("학생별 메시지 수 분포")
                    top_students = student_df.sort_values(by="학생 메시지 수", ascending=False).head(10)
                    chart_data = pd.DataFrame({
                        "학생": top_students["학생명"],
                        "학생 메시지": top_students["학생 메시지 수"],
                        "AI 응답": top_students["AI 응답 수"]
                    })
                    st.bar_chart(chart_data.set_index("학생"))

                    # 등급 분포
                    st.subheader("등급 분포 (기본 분석)")
                    grade_counts = student_df["예상 등급"].value_counts().reset_index()
                    grade_counts.columns = ["등급", "학생 수"]
                    st.bar_chart(grade_counts.set_index("등급"))

                    # 기본 분석 다운로드
                    csv_basic = student_df.to_csv(index=False)
                    st.download_button(
                        label="기본 분석 데이터 다운로드 (CSV)",
                        data=csv_basic,
                        file_name="기본_분석데이터.csv",
                        mime="text/csv"
                    )

                # 키워드 분석은 공통으로 유지
                st.subheader("자주 등장하는 키워드 분석")
                # ... 기존 키워드 분석 코드 그대로 유지 ...
                
            else:
                st.info("분석할 대화 데이터가 없습니다.")
        except Exception as e:
            st.error(f"데이터 분석 중 오류 발생: {str(e)}")
            st.error(f"상세 오류: {traceback.format_exc()}")
            
            
    with admin_tab4:
        st.subheader("전체 데이터 백업")
        
        if st.button("모든 데이터 ZIP으로 다운로드"):
            import zipfile
            import io
            
            # 메모리에 ZIP 파일 생성
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 학생 정보 파일 추가
                if os.path.exists(STUDENTS_FILE):
                    with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
                        zipf.writestr("students.json", f.read())
                
                # 대화 파일 추가
                for filename in os.listdir(CONVERSATIONS_DIR):
                    if filename.endswith('.json'):
                        file_path = os.path.join(CONVERSATIONS_DIR, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            zipf.writestr(f"conversations/{filename}", f.read())
            
            # 다운로드 버튼 표시
            zip_buffer.seek(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="데이터 백업 다운로드",
                data=zip_buffer,
                file_name=f"storyboard_data_backup_{timestamp}.zip",
                mime="application/zip"
            )
            
            st.success("데이터가 성공적으로 압축되었습니다. 다운로드 버튼을 클릭하여 백업 파일을 저장하세요.")
