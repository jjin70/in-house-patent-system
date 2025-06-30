import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "Codes"))
from final_agent import app, tool_selector, PlanExecute

st.set_page_config(page_title="📑 특허 정보 분석 챗봇", layout="wide")

# 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_ready" not in st.session_state:
    st.session_state.eval_ready = False
if "user_input" not in st.session_state:  # 🔧 수정됨
    st.session_state.user_input = None
if "tools" not in st.session_state:  # 🔧 수정됨
    st.session_state.tools = []
if "log" not in st.session_state:
    st.session_state.log = []


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

    weight_mode = st.sidebar.radio("가중치 설정 방식", ["자동", "수동"])
    manual_weights = []

    if weight_mode == "수동":
        st.sidebar.markdown("#### 각 지표의 가중치 (총합 자동 정규화됩니다)")
        # — 여기부터 변경 —
        for indicator in selected_indicators:
            # key를 지정해야 Streamlit이 동적으로 바뀔 때 위젯을 잘 관리합니다
            w = st.sidebar.number_input(
                label=f"{indicator} 가중치",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / len(selected_indicators) if selected_indicators else 0.0,
                key=f"weight_{indicator}"
            )
            manual_weights.append((indicator, w))
        # — 여기까지 변경 —

    if len(selected_indicators) == 5:
        if st.sidebar.button("✅ 지표 설정 완료"):
            st.session_state.eval_ready = True
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
      <p style='font-size: 14px; color:#334155;'>
        📌 이 챗봇은 사용자의 기술 질의에 맞춰  
        <strong>특허 검색</strong>, <strong>특허 동향 분석</strong>,  
        <strong>중요 특허 분석</strong>, <strong>기술 설명서 초안 작성</strong>  
        네 가지 주요 기능을 제공합니다.
      </p>
      <ul style='font-size: 13px; color:#475569; margin-top: 4px;'>
        <li>🔍 <strong>특허 검색</strong> – 사용자의 기술적 질문에 맞는 특허를 검색하고 그 내용을 요약합니다.</li>
        <li>📈 <strong>특허 동향 분석</strong> – 키워드·출원인별 출원 추이를 시각화하여 동향을 파악합니다.</li>
        <li>📊 <strong>중요 특허 분석</strong> – 사용자가 선택한 평가 지표로 특허별 가중치를 계산해 순위를 매깁니다.</li>
        <li>✍️ <strong>기술 설명서 초안 작성</strong> – 기술 개요를 바탕으로 기술 설명서 초안을 작성합니다.</li>
      </ul>
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
new_input = st.chat_input("분석할 기술 문장을 입력하세요...", key="main_chat_input")

# 🔧 사용자 입력이 새로 들어온 경우만 처리
if new_input:
    st.session_state.user_input = new_input
    st.session_state.messages.append({"role": "user", "content": new_input})

    # ✅ 사용자 말풍선 바로 출력
    st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                <div style='background-color: #f0f0f0; color: #000000;
                            padding: 15px; border-radius: 10px;
                            max-width: 33%; word-wrap: break-word;'>
                    <span style='font-size:14px;'>{new_input}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 2) final_agent의 tool_selector를 직접 호출
    state = PlanExecute(input=new_input)
    ts = tool_selector(state)
    st.session_state.tools = ts["tools"]
    # 로그 저장
    st.session_state.log = ts["log"]

    if "patent_evaluator" in st.session_state.tools:
        # st.session_state.messages.append({
        #     "role": "assistant",
        #     "content": "🔧 평가를 진행하려면 왼쪽 사이드바에서 5개의 평가 지표를 선택하고, ‘✅ 지표 설정 완료’ 버튼을 눌러주세요."
        # })
        st.session_state.show_eval_ui = True
    else:
        st.session_state.show_eval_ui = False

    st.session_state.eval_ready = False  # 초기화

# 🔧 툴 조건에 따라 평가 UI 표시
show_evaluation_ui = getattr(st.session_state, "show_eval_ui", False)

# 사이드바 렌더링
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
    with st.spinner("LLM 분석 중…"):
        try:
            # 가중치 계산
            weights_dict = None
            if manual_weights:
                total = sum(w for _, w in manual_weights)
                if total > 0:
                    weights_dict = {k: w / total for k, w in manual_weights}

            # 실행 상태 구성
            state = PlanExecute(
                input=st.session_state.user_input,
                tools=st.session_state.tools,
                sub_queries={},
                results={},
                log=st.session_state.log,
                response=None,
                selected_indicators=selected_indicators,
                indicator_weights=weights_dict
            )
            # 평가 지표 정보가 필요한 경우, PlanExecute 모델에 필드를 추가 후 여기에 설정하세요.

            final = app.invoke(
                state,
                config={"configurable": {
                    "thread_id": "streamlit-session",
                    "checkpoint_ns": "default",
                    "checkpoint_id": "run1"
                }}
            )

            # 결과 출력
            if len(st.session_state.tools) > 1:
                st.markdown("### 💡 종합 결과")
                st.markdown(final["response"])
            #
            # 로그
            # with st.expander("📜 실행 로그"):
            #     for entry in final["log"]:
            #         st.write(entry)

        except Exception as e:
            st.error(f"⚠️ 실행 에러: {e}")

    # 초기화
    st.session_state.eval_ready = False
    st.session_state.user_input = None
    st.session_state.tools = []