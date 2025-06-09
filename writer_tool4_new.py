import re
import streamlit as st  # ✅ 추가
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


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


@tool
def generate_technical_draft(user_input: str) -> str:
    """
    기술 설명서 초안을 작성하고, 누락된 구성요소가 있는 경우 추가 입력을 받아 보완합니다.
    """
    qwen = QwenModel()
    combined_text = user_input

    if "초안" not in user_input:
        return "❌ 초안작성 요청이 아닙니다."

    st.info("📘 다음은 초안작성 도우미 가이드라인입니다. 부족한 항목에 대해서 명시해드리니, 해당 항목에 대해서만 추가 입력해주시면 됩니다.")

    while True:
        draft = qwen.generate_draft(combined_text)
        missing_info = qwen.analyze_missing(draft)

        if not missing_info:
            return f"📄 생성된 기술 설명서 초안:\n\n{draft}"

        st.warning(f"📌 누락된 항목: {', '.join(missing_info)}")
        additional = st.text_input("추가 입력 →", key=f"tool4_missing_{'_'.join(missing_info)}")

        if additional:
            combined_text += " " + additional
        else:
            return f"📄 현재까지의 초안:\n\n{draft}\n\n⚠️ 추가 정보가 입력되지 않아 초안이 완성되지 않았습니다."
