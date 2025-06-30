import pandas as pd
import matplotlib.pyplot as plt
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st

class QwenModel:
    def __init__(self, llm):
        self.llm = llm

    def ask(self, prompt: str) -> str:
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()

class Agent3:
    def __init__(self, csv_path: str, llm, vectorstore_dir: str):
        self.df_all = pd.read_csv(csv_path)
        self.qwen = QwenModel(llm)

        self.embedding = OllamaEmbeddings(model="bge-m3")
        self.vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=self.embedding
        )

    def retrieve_patents_by_rag(self, query: str, top_k=30) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.metadata.get("출원번호") for doc in results]

    def handle(
            self,
            topic_query: str,
            selected_indicators: list[str],
            weight_mode: str = "auto",
            manual_weights: list[float] = None
    ) -> str:
        # 🔍 RAG 기반 특허 검색
        retrieved_ids = self.retrieve_patents_by_rag(topic_query)
        self.df = self.df_all[self.df_all["번호"].isin(retrieved_ids)]

        # 🔍 키워드 필터링
        stopwords = {"및", "관련", "기술", "내용", "시스템", "전기", "특허", "장치", "방법"}
        keywords = [word for word in topic_query.strip().split() if word not in stopwords]

        def match_ratio(text: str) -> float:
            count = sum(1 for k in keywords if k in str(text))
            return count / len(keywords) if keywords else 0

        filtered_df = self.df[self.df["최종키워드"].apply(match_ratio) >= 0.5]
        if not filtered_df.empty:
            self.df = filtered_df

        if self.df.empty:
            return f"❌ '{topic_query}'에 대해 관련 특허가 없습니다."

        # 🧮 가중치 설정
        if weight_mode == "auto":
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        elif weight_mode == "manual":
            if manual_weights is None or len(manual_weights) != 5:
                return "❌ 수동 가중치는 5개 지표에 대해 모두 지정해야 합니다."
            total = sum(manual_weights)
            weights = [w / total for w in manual_weights]
        else:
            return "❌ weight_mode는 'auto' 또는 'manual'이어야 합니다."

        # 📊 점수 계산
        score_df = self.df[selected_indicators].copy()
        weighted_scores = score_df.mul(weights)
        self.df["종합점수"] = weighted_scores.sum(axis=1)

        top_10 = self.df.sort_values("종합점수", ascending=False)[
            ["번호", "출원인", "명칭(번역)", "요약(번역)", "종합점수"] + selected_indicators
            ].head(10)

        result_text = top_10.to_string(index=False)
        prompt = f"""다음은 자연어 쿼리 결과로 생성된 특허 평가 결과입니다:\n\n{result_text}\n\n
    이 결과를 바탕으로 생성된 결과로 알 수 있는 시사점을 한국어로 제시해줘. 이때, 사용자가 꼭 알아야 하는 유의미하고 핵심적인 시사점을 제시해줘야 하며, 특허의 요약을 보고 특허 점수가 높게 나온 특허에 대한 설명도 간단히 제공해줘. 이때 너무 길게 제공하지 말아줘."""

        interpretation = self.qwen.ask(prompt)

        # ✅ Streamlit 출력 (선택 사항)
        st.markdown("### 📊 중요 특허 평가 결과")
        st.markdown(top_10.to_markdown(index=False))
        st.markdown("### 🧠 시사점 요약")
        st.markdown(interpretation)

        # ✅ 출력 문자열 반환
        return result_text + "\n\n" + interpretation
