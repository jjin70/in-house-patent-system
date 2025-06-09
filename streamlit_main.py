import streamlit as st
from final_agent import app, PlanExecute
from uuid import uuid4

st.set_page_config(page_title="특허 분석 챗봇", layout="wide")

st.title("📌 특허 분석 챗봇")
st.markdown("질문을 입력하면 적절한 분석 도구를 자동 선택해 결과를 종합해드립니다.")

user_input = st.text_area("질문을 입력하세요", height=150)

if st.button("분석 시작"):
    if not user_input.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("🔍 분석 중입니다..."):
            try:
                state = PlanExecute(
                    input=user_input,
                    tools=[],
                    sub_queries={},
                    results={},
                    response=None,
                    log=[],
                    selected_indicator_indexes=[],
                    weight_mode="auto",
                    manual_weights=None
                )                # thread_id 추가
                final_state = app.invoke(state, config={"configurable": {"thread_id": str(uuid4())}})

                st.success("✅ 분석 완료")
                st.markdown("### 🔎 최종 응답")
                st.markdown(final_state.response)

                with st.expander("📄 로그 보기"):
                    for log in final_state.log:
                        st.markdown(f"- {log}")

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

            # 실행 후 초기화 안 되면 다음 쿼리에 영향을 줌
            st.session_state.eval_ready = False
            st.session_state.user_input = None
            st.session_state.tools = []
