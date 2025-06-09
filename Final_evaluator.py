import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
from typing import List
import streamlit as st  # ✅ Streamlit 추가

class QwenModel:
    def __init__(self, llm):
        self.llm = llm

    def ask(self, prompt: str) -> str:
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()

class Agent3:
    def __init__(self, csv_path: str, llm):
        self.df = pd.read_csv(csv_path)
        self.qwen = QwenModel(llm)

    def filter_by_keywords(self, topic_query: str) -> bool:
        stopwords = {"및", "관련", "특허", "평가", "기술", "대한"}
        keywords = [word for word in topic_query.split() if word not in stopwords]

        if not keywords:
            return False

        def keyword_match_ratio(text: str) -> float:
            count = sum(1 for k in keywords if k in text)
            return count / len(keywords) if keywords else 0

        self.df["_매칭비율"] = self.df["요약(번역)"].fillna("").apply(keyword_match_ratio)
        self.df = self.df[self.df["_매칭비율"] >= 0.5].drop(columns=["_매칭비율"])

        return not self.df.empty

    def handle(
        self,
        topic_query: str,
        selected_indicator_indexes: List[int] = [1, 2, 3, 4, 5],
        weight_mode: str = "auto",  # "auto" or "manual"
        manual_weights: List[float] = None
    ):
        if not self.filter_by_keywords(topic_query):
            return "관련 특허가 없어 평가할 수 없습니다."

        indicators = [
            "참여 발명자 수준", "기술 영향력", "기술 지속성", "시장성",
            "기술 집중도", "신규성", "권리의 광역성", "권리의 완전성"
        ]

        selected_indicators = [indicators[i - 1] for i in selected_indicator_indexes]

        if weight_mode == "auto":
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        elif weight_mode == "manual":
            if manual_weights is None or len(manual_weights) != 5:
                return "❌ 수동 가중치는 반드시 5개를 입력해야 합니다."
            total = sum(manual_weights)
            weights = [w / total for w in manual_weights]
        else:
            return "❌ weight_mode는 'auto' 또는 'manual'이어야 합니다."

        score_df = self.df[selected_indicators].copy()
        weighted_scores = score_df.mul(weights)
        self.df["종합점수"] = weighted_scores.sum(axis=1)

        top_10 = self.df.sort_values("종합점수", ascending=False)[
            ["번호", "출원인", "명칭(번역)", "요약(번역)", "종합점수"] + selected_indicators
        ].head(10)

        result_text = top_10.to_markdown(index=False)
        st.markdown("### 📊 중요 특허 평가 결과")
        st.markdown(result_text)

        prompt1 = f"""다음은 자연어 쿼리 결과로 생성된 특허 평가 결과입니다:\n\n{top_10.to_string(index=False)}\n\n
이 결과를 바탕으로 생성된 결과로 알 수 있는 시사점을 한국어로 제시해줘. 이때, 사용자가 꼭 알아야 하는 유의미하고 핵심적인 시사점을 제시해줘야 하며, 특허의 요약을 보고 특허 점수가 높게 나온 특허에 대한 설명도 간단히 제공해줘. 이때 너무 길게 제공하지 말아줘."""

        interpretation = self.qwen.ask(prompt1)

        # ✅ 여기서도 바로 출력
        st.markdown("### 🧠 시사점 요약")
        st.markdown(interpretation)

        return None
