import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
import streamlit as st  # ✅ 추가

# plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

class QwenModel:
    def __init__(self, llm):
        self.llm = llm

    def ask(self, prompt: str) -> str:
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()

class KeywordAnalyzer:
    def __init__(self, csv_path: str, llm):
        self.df = pd.read_csv(csv_path)
        self.qwen = QwenModel(llm)

        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

        if "출원일" in self.df.columns:
            self.df["출원일"] = pd.to_datetime(self.df["출원일"].astype(str) + "-01-01", errors="coerce")
            self.df["출원연도"] = self.df["출원일"].dt.year

    def extract_keyword_and_year(self, text):
        range_match = re.search(r"(\d{4})\D*(\d{4})", text)
        if range_match:
            year_range = (int(range_match.group(1)), int(range_match.group(2)))
        elif "이후" in text or "부터" in text:
            single_year_match = re.search(r"(\d{4})", text)
            year_range = (int(single_year_match.group(1)), 2100) if single_year_match else None
        elif "이전" in text or "까지" in text:
            single_year_match = re.search(r"(\d{4})", text)
            year_range = (1900, int(single_year_match.group(1))) if single_year_match else None
        else:
            year_range = None

        keyword = re.sub(r"\d{4}.*", "", text)
        keyword = re.sub(r"\b\w*의\b", "", keyword)
        noise_words = ["기술", "과", "된", "관련", "연도별", "출원", "특허", "동향", "건수", "현황", "추이", "이후", "부터", "이전", "까지"]
        for word in noise_words:
            keyword = keyword.replace(word, "")
        return keyword.strip(), year_range

    def corporate(self, keyword: str, year_range=None):
        matches = self.df[self.df["출원인"].str.contains(keyword, na=False, case=False)]
        return self._plot_and_interpret(matches, keyword, "출원인", year_range)

    def tech(self, keyword: str, year_range=None):
        matches = self.df[self.df["요약(번역)"].str.contains(keyword, na=False, case=False)]
        return self._plot_and_interpret(matches, keyword, "기술", year_range)

    def _plot_and_interpret(self, matches, keyword, keyword_type, year_range):
        output_text = ""

        if matches.empty:
            output_text += f"❌ '{keyword}' 관련 특허가 없습니다.\n"
            return output_text

        summary = matches.groupby("출원연도").size().reset_index(name="출원 건수")
        if year_range:
            start, end = year_range
            summary = summary[(summary["출원연도"] >= start) & (summary["출원연도"] <= end)]

        if summary.empty:
            output_text += f"❌ 주어진 연도 범위 내에 '{keyword}' 관련 특허가 없습니다.\n"
            return output_text

        # ✅ Streamlit용 그래프
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.bar(summary["출원연도"], summary["출원 건수"], color="navy")
        ax.set_xlabel("출원연도", fontsize=5)
        ax.set_ylabel("출원 건수", fontsize=5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        try:
            prompt = f"""다음은 '{keyword}'({keyword_type}) 관련 연도별 특허 출원 건수입니다:\n
{summary.to_string(index=False)}

이를 해석해줄 때 꼭 한국어로 답변해야 하며, 전체적인 그래프의 흐름을 기술 동향과 연결해 간결하게 설명해주세요."""
            interpretation = self.qwen.ask(prompt)

            st.markdown("### 📊 연도별 특허 출원 데이터")
            st.markdown(summary.to_markdown(index=False))

            st.markdown("### 🧠 트렌드 해석 결과")
            st.markdown(interpretation)

        except Exception as e:
            st.error(f"⚠️ LLM 해석 실패: {e}")

    def run(self, query: str):
        keyword, year_range = self.extract_keyword_and_year(query)

        if self.df["출원인"].astype(str).str.contains(keyword, case=False, na=False).any():
            self.corporate(keyword, year_range)
        elif self.df["요약(번역)"].astype(str).str.contains(keyword, case=False, na=False).any():
            self.tech(keyword, year_range)
        else:
            st.warning("❌ 키워드 매칭 실패: '출원인'이나 '요약(번역)'에 해당 키워드가 없습니다.")  
            