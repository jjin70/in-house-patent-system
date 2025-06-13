import re
import streamlit as st  # ✅ 추가
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

class QwenModel:
    def __init__(self, model_name="qwen2.5:7b", temperature=0.0):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.last_response = ""

    def generate_draft(self, content: str) -> str:
        prompt = self.load_prompt_template().format(text=content)
        for retry in range(5):
            response = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if not self.contains_chinese(response):
                return response
            st.warning(f"⚠️ 한자 포함 감지. 재시도 {retry + 1}/5...")  # ✅ Streamlit 출력
        return response

    @staticmethod
    def contains_chinese(text):
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    @staticmethod
    def load_prompt_template():
        with open("/Users/heejinyang/python/streamlit/Codes/prompt_new.txt", "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def analyze_missing(response: str) -> list[str]:
        missing = []
        if "해결하고자 하는 문제가 명시되어 있지 않음" in response:
            missing.append("해결하고자 하는 문제")
        if "발명 과정이 명시되어 있지 않음" in response:
            missing.append("발명 과정")
        if "발명 내용이 명시되어 있지 않음" in response:
            missing.append("발명 내용")
        if "기대 효과가 명시되어 있지 않음" in response:
            missing.append("기대 효과")
        return missing

# @tool
def generate_technical_draft(user_input: str) -> str:
    qwen = QwenModel()

    # 1️⃣ 최초 상태 초기화
    if "tool4_base_input" not in st.session_state:
        st.session_state.tool4_base_input = ""
        st.session_state.tool4_combined_input = ""
        st.session_state.tool4_additional_inputs = []
        st.session_state.tool4_retry = 0
        st.session_state.tool4_draft = ""
        st.session_state.tool4_missing = []
        st.session_state.tool4_done = False

    # 2️⃣ 기본 기술 설명이 없으면 입력받기
    if not st.session_state.tool4_base_input:
        st.info("✏️ 기술 문장 초안을 작성하기 위해 먼저 기술 설명을 입력해 주세요.")
        base = st.text_area("", key="tool4_initial_input", height=120, placeholder="🔧 기술에 대한 설명을 작성해 주세요.")

        if st.button("🚀 초안 생성 시작"):
            if base.strip():
                st.session_state.tool4_base_input = base.strip()
                st.session_state.tool4_combined_input = base.strip()
                st.rerun()
            else:
                st.warning("⚠️ 입력이 비어 있습니다.")
        return

    # 3️⃣ 기술 설명은 항상 위에 표시
    st.markdown("#### 📌 기술 설명 (기본 입력)")
    st.info(st.session_state.tool4_base_input)

    # 4️⃣ 이전 보완 입력 표시
    if st.session_state.tool4_additional_inputs:
        st.markdown("#### ✍️ 추가 입력된 기술 설명")
        st.info("\n\n".join(st.session_state.tool4_additional_inputs))

    # 5️⃣ 초안이 없으면 생성
    if not st.session_state.tool4_draft:
        draft = qwen.generate_draft(st.session_state.tool4_combined_input)
        missing = qwen.analyze_missing(draft)

        st.session_state.tool4_draft = draft
        st.session_state.tool4_missing = missing
        st.session_state.tool4_done = not missing

    # 6️⃣ 초안 완성 시 종료
    if st.session_state.tool4_done:
        st.markdown("### 📄 기술 설명서 초안")
        st.success("✅ 모든 항목이 포함된 최종 초안입니다.")
        st.text(st.session_state.tool4_draft)
        return

    # 7️⃣ 누락 항목 보완 입력
    st.warning(f"📌 누락된 항목: {', '.join(st.session_state.tool4_missing)}")

    additional = st.text_area(
        "🔧 누락 항목에 대한 보완 설명을 작성해주세요",
        key=f"tool4_additional_input_{st.session_state.tool4_retry}",
        height=100
    )

    if st.button("🔄 보완 내용으로 초안 다시 생성"):
        if additional.strip():
            st.session_state.tool4_additional_inputs.append(additional.strip())
            st.session_state.tool4_combined_input += " " + additional.strip()
            st.session_state.tool4_draft = ""
            st.session_state.tool4_missing = []
            st.session_state.tool4_done = False
            st.session_state.tool4_retry += 1
            st.rerun()
        else:
            st.warning("⚠️ 입력이 비어 있습니다.")
