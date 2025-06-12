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

    # 최초 입력 또는 새 입력일 경우 상태 초기화
    if "tool4_base_input" not in st.session_state or user_input != st.session_state.tool4_base_input:
        st.session_state.tool4_base_input = user_input  # 최초 입력 고정
        st.session_state.tool4_combined_input = user_input  # 누적 입력
        st.session_state.tool4_draft = ""
        st.session_state.tool4_missing = []
        st.session_state.tool4_done = False
        st.session_state.tool4_retry = 0

    # 초안이 없으면 생성
    if not st.session_state.tool4_draft:
        with st.spinner("초안 생성 중..."):
            draft = qwen.generate_draft(st.session_state.tool4_combined_input)
            missing = qwen.analyze_missing(draft)

            st.session_state.tool4_draft = draft
            st.session_state.tool4_missing = missing
            st.session_state.tool4_done = not missing

    # 누락 항목 없이 완성된 경우
    if st.session_state.tool4_done:
        st.success("✅ 모든 항목이 포함된 최종 초안입니다.")
        st.markdown(f"📄 **기술 설명서 초안**\n\n{st.session_state.tool4_draft}")
        return

    # 누락 항목이 있을 경우 보완 입력 받기
    if st.session_state.tool4_missing:
        st.warning(f"📌 누락된 항목: {', '.join(st.session_state.tool4_missing)}")

        additional = st.text_area(
            "🔧 누락 항목에 대한 보완 설명을 작성해주세요",
            key=f"tool4_additional_input_{st.session_state.tool4_retry}",
            height=100,
            placeholder="예: 해당 기술은 고온 환경에서도 배터리 성능을 유지하도록 설계되었습니다..."
        )

        # 버튼 클릭 여부 상태 키
        regen_key = f"tool4_regen_pressed_{st.session_state.tool4_retry}"

        # 버튼이 처음 눌렸는지 상태 등록
        if regen_key not in st.session_state:
            st.session_state[regen_key] = False

        # 버튼 눌림 처리
        if st.button("🔄 보완 내용으로 초안 다시 생성", key=f"regen_button_{st.session_state.tool4_retry}"):
            if additional.strip():
                # 상태 업데이트
                st.session_state.tool4_combined_input += " " + additional.strip()
                st.session_state.tool4_draft = ""
                st.session_state.tool4_missing = []
                st.session_state.tool4_done = False
                st.session_state.tool4_retry += 1
                st.session_state[regen_key] = True  # ✅ rerun 트리거용 상태 저장
                st.rerun()
            else:
                st.warning("⚠️ 입력이 비어 있습니다.")
