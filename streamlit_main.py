import streamlit as st
import sys
import os
import pandas as pd
import chromadb
import uuid
import json
from Final_evaluator import Agent3
from langchain_community.chat_models import ChatOllama

# ✅ llm 정의
llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)

sys.path.append(os.path.join(os.getcwd(), "Codes"))
from final_agent import app, tool_selector_chain, extract_json_from_text

# CSV 불러오기
csv_path = "/Users/heejinyang/python/streamlit/csv/0527_cleaning_processing_ver1.csv"
df = pd.read_csv(csv_path)

# Chroma DB 연결
client = chromadb.PersistentClient(path="/Users/heejinyang/python/streamlit/chroma_db_streamlit")
collection = client.get_or_create_collection(name="my_collection")

# 텍스트 파일
with open("/Users/heejinyang/python/streamlit/Streamlit_patent_list/Streamlit_patents.txt", "r", encoding="utf-8") as f:
    patent_list = f.read().splitlines()

st.set_page_config(page_title="📑 특허 정보 분석 챗봇", layout="wide") 

# 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_ready" not in st.session_state:
    st.session_state.eval_ready = False
if "user_input" not in st.session_state:  # 🔧 수정됨
    st.session_state.user_input = None
if "tools" not in st.session_state:       # 🔧 수정됨
    st.session_state.tools = []

# tool selector
def run_tool_selector(user_input: str):
    response = tool_selector_chain.invoke({"query": user_input})
    raw = response.content.strip()
    print(f"[TOOL_SELECTOR 응답 원문]:\n{raw}")  # 🔧 추가
    json_text = extract_json_from_text(raw)
    parsed = json.loads(json_text)
    print(f"[TOOL_SELECTOR 파싱 결과]:\n{parsed}")  # 🔧 추가
    return parsed["tools"]

def show_sidebar_evaluation_ui():
    indicator_names = [
        "참여 발명자 수준", "기술 영향력", "기술 지속성", "시장성",
        "기술 집중도", "신규성", "권리의 광역성", "권리의 완전성"
    ]

    st.sidebar.markdown("### 📊 평가 지표 선택")
    selected_indicators = st.sidebar.multiselect(
        "중요한 평가 지표 5개 선택",
        indicator_names,
        default=[]
    )

    weight_mode = st.sidebar.radio("가중치 설정 방식", ["자동", "수동"]) # TODO 수동일 때 가중치1,2,3,4 이렇게 띄우면 안 되고 뭔지 알려줘야할듯?
    manual_weights = []

    if weight_mode == "수동":
        st.sidebar.markdown("#### 각 지표의 가중치 (총합 자동 정규화됩니다)")
        for i in range(5):
            w = st.sidebar.number_input(f"가중치 {i+1}", min_value=0.0, max_value=100.0, value=20.0)
            manual_weights.append(w)

    if len(selected_indicators) == 5:
        if st.sidebar.button("✅ 지표 설정 완료"):
            st.session_state.eval_ready = True  # 🔧 수정됨
    else:
        st.sidebar.warning("⚠️ 평가 지표는 반드시 5개를 선택해야 합니다.")

    return indicator_names, selected_indicators, weight_mode, manual_weights

# 헤더
st.markdown("""
    <h2 style='
        color: #1c3162;
        font: sans-serif;
        font-weight: 600;
        font-size: 30px;
        margin-bottom: 8px;
    '>📑 특허 정보 분석 챗봇</h2>
    <div style='padding-left: 20px;'>
        <p style='font-size: 14px; font-weight: 600; color:#334155;'>
            뭔가 설명을 넣어야할듯
        </p>
    </div>
""", unsafe_allow_html=True)

# 이전 메시지 출력
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    if is_user:
        st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='background-color: #f0f0f0; color: #000000;
                            padding: 15px; border-radius: 10px;
                            max-width: 33%; word-wrap: break-word;'>
                    <span style='font-size:14px;'>{msg["content"]}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='
                display: flex; 
                justify-content: flex-start; 
                margin: 20px 0;
                padding-right: 20%;
            '>
                <div style='
                    font-size: 14px; 
                    line-height: 1.6;
                    color: #1E293B;
                    white-space: pre-wrap;
                '>
                    {msg["content"]}
                </div>
            </div>
        """, unsafe_allow_html=True)

# 채팅 입력
new_input = st.chat_input("분석할 기술 문장을 입력하세요...", key="main_chat_input") # TODO 바꿔야 하지 않을까????

# 🔧 사용자 입력이 새로 들어온 경우만 처리
if new_input:
    # 1️⃣ 사용자 메시지 세션 저장
    st.session_state.user_input = new_input
    st.session_state.messages.append({"role": "user", "content": new_input})
    
    # 2️⃣ 사용자 입력 즉시 출력
    st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
            <div style='background-color: #f0f0f0; color: #000000;
                        padding: 15px; border-radius: 10px;
                        max-width: 33%; word-wrap: break-word;'>
                <span style='font-size:14px;'>{new_input}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3️⃣ spinner 안에서 LLM 처리
    with st.spinner("서브 쿼리 분석 중..."):
        st.session_state.tools = run_tool_selector(new_input)

    st.session_state.eval_ready = False

# 🔧 툴 조건에 따라 평가 UI 표시
show_evaluation_ui = "patent_evaluator" in st.session_state.tools
run_analysis = False

if st.session_state.user_input:
    if show_evaluation_ui:
        indicator_names, selected_indicators, weight_mode, manual_weights = show_sidebar_evaluation_ui()
        run_analysis = st.session_state.eval_ready
    else:
        indicator_names = []
        selected_indicators = []
        weight_mode = "auto"
        manual_weights = None
        run_analysis = True

# 🔧 분석 조건 만족 시 실행
if st.session_state.user_input and run_analysis:
    tools = st.session_state.tools

    if "patent_searcher" in tools:
        try:
            with st.spinner("🔍 특허 검색 및 요약 분석 중..."):
                result = app.invoke(
                    {
                        "input": st.session_state.user_input,
                        "selected_indicator_indexes": [indicator_names.index(i) + 1 for i in selected_indicators] if selected_indicators else [],
                        "weight_mode": "manual" if weight_mode == "수동" else "auto",
                        "manual_weights": manual_weights if weight_mode == "수동" else None,
                    },
                    config={"configurable": {"thread_id": str(uuid.uuid4())}}
                )

                analysis_text = result.get("results", {}).get("patent_searcher", "")
                summary_text = result.get("response", "")

                if analysis_text.startswith("[서브 쿼리 1]"):
                    analysis_text = "\n".join(analysis_text.split("\n")[1:]).strip()

                st.markdown("<h3>📘 특허 검색 및 요약 정보</h3>", unsafe_allow_html=True)
                if "📘" in analysis_text:
                    analysis_text = analysis_text.replace("📘 특허 검색 및 요약 정보:", "")
                st.markdown(f"<div style='font-size:16px;'>{analysis_text}</div>", unsafe_allow_html=True)

                if "📄" in analysis_text:
                    st.markdown("<h3 style='margin-top:32px;'>📄 Selector에서 제외된 특허 요약 정보</h3>", unsafe_allow_html=True)

                st.markdown("---", unsafe_allow_html=True)
                st.markdown("<h3>🧠 종합 요약</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:16px;'>{summary_text}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ 특허 검색 도중 오류 발생: {e}")

    if "patent_evaluator" in tools:
        try:
            with st.spinner("📊 특허 평가 중..."):
                evaluator = Agent3(csv_path=csv_path, llm=llm)
                interpretation = evaluator.handle(
                    topic_query=st.session_state.user_input,
                    selected_indicator_indexes=[indicator_names.index(i) + 1 for i in selected_indicators],
                    weight_mode="manual" if weight_mode == "수동" else "auto",
                    manual_weights=manual_weights if weight_mode == "수동" else None
                )
        except Exception as e:
            st.error(f"⚠️ 특허 평가 도중 오류 발생: {e}")
    
    if "patent_trend_analyzer" in tools:
        try:
            from Final_Trend import KeywordAnalyzer  # ✅ 모듈이 이 이름이면
            trend = KeywordAnalyzer(csv_path=csv_path, llm=llm)
            with st.spinner("📈 특허 트렌드 분석 중..."):
                trend.run(st.session_state.user_input)
        except Exception as e:
            st.error(f"⚠️ 트렌드 분석 도중 오류 발생: {e}")

    if "tech_writer" in tools:
        try:
            from writer_tool4_new import generate_technical_draft  # ✅ 정확한 함수명으로 대체
            with st.spinner("📝 기술 설명서 초안 작성 중..."):
                draft = generate_technical_draft(st.session_state.user_input)
                st.markdown("### ✍️ 기술 설명서 초안")
                st.markdown(draft)
        except Exception as e:
            st.error(f"⚠️ 기술 설명서 작성 중 오류 발생: {e}")

    # 상태 초기화
    st.session_state.eval_ready = False
    st.session_state.user_input = None
    st.session_state.tools = []
