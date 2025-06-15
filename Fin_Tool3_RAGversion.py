import pandas as pd
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Qwen 모델 Wrapper
class QwenModel:
    def __init__(self, llm):
        self.llm = llm

    def ask(self, prompt: str) -> str:
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()

# 최종 통합 Agent
class RAGBasedAgent3:
    def __init__(self, csv_path: str, llm, vectorstore_dir: str = "Tool3_DB"):
        self.df_all = pd.read_csv(csv_path)
        self.qwen = QwenModel(llm)

        # RAG용 임베딩 + vectorstore
        self.embedding = OllamaEmbeddings(model="bge-m3")
        self.vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=self.embedding
        )

    def retrieve_patents_by_rag(self, query: str, top_k=30) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.metadata.get("출원번호") for doc in results]

    def handle(self, topic_query: str):
        retrieved_ids = self.retrieve_patents_by_rag(topic_query)
        self.df = self.df_all[self.df_all["번호"].isin(retrieved_ids)]

        # ✅ 키워드 기반 필터링 (불용어 제거 후 0.5 이상 비율로 포함된 경우 유지)
        stopwords = {"및", "관련", "기술", "내용", "시스템", "전기", "특허", "장치", "방법"}
        keywords = [word for word in topic_query.strip().split() if word not in stopwords]

        def match_ratio(text: str) -> float:
            count = sum(1 for k in keywords if k in str(text))
            return count / len(keywords) if keywords else 0

        filtered_df = self.df[self.df["최종키워드"].apply(match_ratio) >= 0.5] # 0.5로해야 그래도 filtering느낌이 듬 -> 이걸로 하나도 안걸러지면 RAG결과만 반영되게 구성함.

        if not filtered_df.empty:
            #print(f"🔎 키워드 포함 비율 0.5 이상 결과 {len(filtered_df)}건 → 이를 사용합니다.")
            self.df = filtered_df

        if self.df.empty:
            return f"❌ '{topic_query}'에 대해 관련 특허가 없습니다."

        print(f"✅ 총 {len(self.df)}개의 특허가 최종 후보로 선택되었습니다.")

        print("\n경쟁사 특허 평가의 지표는 다음과 같습니다:")
        print("1. 참여 발명자 수준\n2. 기술 영향력\n3. 기술 지속성\n4. 시장성\n5. 기술 집중도\n6. 신규성\n7. 권리의 광역성\n8. 권리의 완전성")

        indicators = ["참여 발명자 수준", "기술 영향력", "기술 지속성", "시장성", "기술 집중도", "신규성", "권리의 광역성", "권리의 완전성"]

        while True:
            indicator_input = input("\n중요하다고 생각하는 5개 지표의 번호를 중요한 순서대로 입력하세요 (1~8 범위, 띄어쓰기 구분): ")
            parts = indicator_input.split()

            if len(parts) != 5 or any(not p.isdigit() for p in parts):
                print("\n❗ 숫자 5개를 띄어쓰기 구분으로 입력해주세요.")
                continue

            nums = [int(p) for p in parts]
            if any(n < 1 or n > 8 for n in nums):
                print("\n❗ 1부터 8 사이의 숫자만 입력해주세요.")
                continue

            selected_indicators = [indicators[n - 1] for n in nums]
            break

        while True:
            mode = input("\n가중치 설정 방식을 선택하세요: 1. 우선순위 기반 자동, 2. 수동 입력: ")
            if mode.strip() == "1":
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
                print(f"\n🔧 자동 적용된 가중치: {weights}")
                break
            elif mode.strip() == "2":
                while True:
                    importance_input = input("\n5개 지표의 중요도를 입력하세요 (띄어쓰기 구분): ")
                    try:
                        importances = [float(w) for w in importance_input.split()]
                        if len(importances) != 5:
                            raise ValueError("갯수 오류")
                        total = sum(importances)
                        weights = [imp / total for imp in importances]
                        break
                    except:
                        print("\n❗ 5개의 숫자만 정확히 입력해주세요.")
                break
            else:
                print("\n❗ 1 또는 2 중에서 선택해주세요.")

        score_df = self.df[selected_indicators].copy()
        weighted_scores = score_df.mul(weights)
        self.df["종합점수"] = weighted_scores.sum(axis=1)

        top_10 = self.df.sort_values("종합점수", ascending=False)[["번호", "출원인", "명칭(번역)", "요약(번역)", "종합점수"] + selected_indicators].head(10)
        print("\n📊 종합점수 기준 상위 10개 특허:")
        print(top_10.reset_index(drop=True))

        result_text = top_10.to_string(index=False)

        prompt1 = f"""다음은 자연어 쿼리 결과로 생성된 특허 평가 결과입니다:\n\n{result_text}\n\n
이 결과를 바탕으로 생성된 결과로 알 수 있는 시사점을 한국어로 제시해줘. 이때, 사용자가 꼭 알아야 하는 유의미하고 핵심적인 시사점을 제시해줘야 하며, 특허의 요약을 보고 특허 점수가 높게 나온 특허에 대한 설명도 간단히 제공해줘. 이때 너무 길게 제공하지 말아줘 """
        interpretation = self.qwen.ask(prompt1)
        print("\n📌 중요 특허 요약:")
        print(interpretation)
        return interpretation
