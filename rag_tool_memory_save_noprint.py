import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict
import json
import re
import streamlit as st
from difflib import SequenceMatcher
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.retrievers import BM25Retriever


def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def ensemble_with_normalized_scores(docs1, docs2, weight1=0.6, weight2=0.4, top_k=10):
    def safe_uid(doc):
        app_no = doc.metadata.get("출원번호")
        chunk_id = doc.metadata.get("chunk_id")
        if app_no is None or chunk_id is None:
            return None
        return (str(app_no), str(chunk_id))  # 문자열로 통일

    scores1 = {}
    scores2 = {}
    doc_map = {}

    for doc in docs1:
        uid = safe_uid(doc)
        if uid is not None:
            scores1[uid] = doc.metadata.get("score", 0)
            doc_map[uid] = doc

    for doc in docs2:
        uid = safe_uid(doc)
        if uid is not None:
            scores2[uid] = doc.metadata.get("score", 0)
            doc_map[uid] = doc

    uids = list(set(scores1.keys()).union(scores2.keys()))
    score_list1 = normalize([scores1.get(uid, 0) for uid in uids])
    score_list2 = normalize([scores2.get(uid, 0) for uid in uids])
    final_scores = [weight1 * s1 + weight2 * s2 for s1, s2 in zip(score_list1, score_list2)]

    # 정렬 수행
    ranked = sorted(zip(final_scores, uids), reverse=True)
    return [doc_map[uid] for _, uid in ranked[:top_k]]


# 사후 필터링 함수
def apply_post_filter(docs: List[Document], parsed_filter: Dict) -> List[Document]:
    def is_similar(a: str, b: str, threshold: float = 0.4) -> bool:
        return SequenceMatcher(None, a,
                               b).ratio() >= threshold  # Threshold는 현대자동차를 현대차, 현대기아차, 현대 와같은 여러 기업명에 잘 적용되도록 조절하는 함수임

    filtered_docs = []
    for doc in docs:
        meta = doc.metadata
        keep = True

        if "출원인" in parsed_filter and isinstance(parsed_filter["출원인"], list) and parsed_filter["출원인"]:
            meta_applicant = meta.get("출원인", "")
            if meta_applicant == "특정 기업":
                keep = True
            elif not any(is_similar(applicant, meta_applicant) for applicant in parsed_filter["출원인"]):
                keep = False

        if "출원일자" in parsed_filter and isinstance(parsed_filter["출원일자"], dict):
            doc_date = meta.get("출원일자", None)
            if isinstance(doc_date, str):
                try:
                    doc_date = int(doc_date)
                except ValueError:
                    keep = False

            if isinstance(doc_date, int):
                for condition, target_value in parsed_filter["출원일자"].items():
                    if condition == "$gte" and doc_date < target_value:
                        keep = False
                    if condition == "$lte" and doc_date > target_value:
                        keep = False

        if keep:
            filtered_docs.append(doc)

    return filtered_docs


def extract_json_from_text(text: str) -> str:
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    raise ValueError("JSON 포맷을 찾을 수 없습니다.")


def group_docs_by_application_number(docs: List[Document]) -> Dict[str, List[Document]]:
    grouped = defaultdict(list)
    for doc in docs:
        app_num = doc.metadata.get("출원번호", "UNKNOWN")
        grouped[app_num].append(doc)
    return grouped


def expand_documents_by_application_number(
        docs: List[Document],
        vectorstore: Chroma
) -> Tuple[List[Document], pd.DataFrame]:
    summarization_sections = {"기술배경", "발명의효과"}
    structured_sections = {"요약", "독립청구항", "기술과제"}

    app_nums = list(set(doc.metadata.get("출원번호") for doc in docs if "출원번호" in doc.metadata))
    if not app_nums:
        return docs, pd.DataFrame(columns=["출원번호", "section", "내용"])

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"출원번호": {"$in": app_nums}}, "k": 20}
    )
    expanded = retriever.invoke("")

    # ✅ 중복 제거
    seen = set()
    deduped = []
    for doc in docs + expanded:
        uid = (doc.metadata.get("출원번호", ""), doc.metadata.get("chunk_id", ""))
        if uid not in seen:
            seen.add(uid)
            deduped.append(doc)

    summary_docs = []
    structured_records = []

    for doc in deduped:
        app_num = doc.metadata.get("출원번호", "UNKNOWN")
        section = doc.metadata.get("section", "기타")
        content = doc.page_content.strip()

        if section in summarization_sections:
            summary_docs.append(doc)
        elif section in structured_sections:
            structured_records.append({
                "출원번호": app_num,
                "section": section,
                "내용": content
            })

    # ✅ 구조화 정보 → DataFrame 변환
    structured_info_df = pd.DataFrame(structured_records)
    return summary_docs, structured_info_df


def retrieve_structured_sections_by_appnums(app_nums: List[str], vectorstore: Chroma) -> pd.DataFrame:
    """
    선택된 출원번호별로 '요약', '기술과제', '독립청구항'을 각각 별도로 리트리브하여 통합
    """
    structured_sections = {"요약", "기술과제", "독립청구항"}
    rows = []

    for app_num in app_nums:
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"출원번호": app_num}, "k": 30}
        )
        docs = retriever.invoke("")

        for doc in docs:
            section = doc.metadata.get("section", "")
            if section in structured_sections:
                rows.append({
                    "출원번호": doc.metadata.get("출원번호", "N/A"),
                    "출원인": doc.metadata.get("출원인", "N/A"),
                    "section": section,
                    "내용": doc.page_content.strip()
                })

    df = pd.DataFrame(rows)
    df["section"] = pd.Categorical(df["section"], categories=["기술과제", "요약", "독립청구항"], ordered=True)
    df.sort_values(["출원번호", "section"], inplace=True)
    return df


def generate_advanced_summary(selected_docs, user_query, llm):
    # ✅ 출원번호 기준 그룹핑 + 모든 section 저장
    grouped = defaultdict(lambda: defaultdict(str))  # app_num → section → content

    for doc in selected_docs:
        app_num = doc.metadata.get("출원번호", "UNKNOWN")
        section = doc.metadata.get("section", "기타")
        content = doc.page_content.strip()
        # ✅ 이제 모든 section 포함 (중복 방지)
        if section not in grouped[app_num]:
            grouped[app_num][section] = content  # 첫 등장만 저장

    # ✅ Context 구성 (독립청구항 포함)
    chunks = []
    for i, (app_num, sections) in enumerate(grouped.items(), start=1):
        example_doc = next(doc for doc in selected_docs if doc.metadata.get("출원번호") == app_num)
        title = example_doc.metadata.get("발명의 명칭", "N/A")
        applicant = example_doc.metadata.get("출원인", "N/A")

        doc_block = f"출원번호: {app_num}\n출원인: {applicant}\n발명의 명칭: {title}"
        for section, content in sections.items():
            doc_block += f"\n[{section}]\n{content}"
        chunks.append(doc_block)

    context = "\n---------------------\n".join(chunks)

    # ✅ 프롬프트 구성
    system_prompt = """
당신은 전기차 기술 분야에 대해 전문적인 분석과 답변을 제공하는 특허 전문가입니다.
Context는 특허번호별로 질문과 관계 있는 해당 특허의 정보가 담겨있습니다.
이에 대해 다음과 같은 구조로 답변을 작성하되, 특허가 등장하게 된 배경과 어떻게 기존의 문제를 해결하고자 했는지, 그리고 이를 통한 발명의 효과를 중점적으로 설명해주세요.

[작성 지침]
- 사용자 질의와 관련있는 특허별 설명에는 반드시 "출원번호", "출원인", "발명의 명칭"을 반드시 포함하세요.(임의로 지어내지 마세요.)
- 자연스러운 한국어 문어체로 작성하세요.
- 특허 설명은 요약적이고 간결하게, 흐름은 부드럽게 연결하세요.
- 이때, 서로 다른 특허가 뽑혔다면, 그 특허에 대한 비교 분석을 수행해도 됩니다.
"""
    user_prompt = """
아래 Context에는 사용자의 질문을 해결하기 위해 선택된 특허 정보들이 포함되어 있습니다.
답변 작성 시 반드시 Context에 포함된 특허들의 모든 '출원번호', '출원인', '발명의 명칭'을 정확히 인용하고, 특허별 기술 내용을 자연스럽게 활용하세요.
각 특허의 내용은 Context에서 '---------------------'로 줄바꿈되어 구분되어있습니다.

Context:
---------------------
{context}
---------------------

질문:
{question}

==> Context 기반으로 답변을 작성하세요. 논리적이고 자연스러운 문단을 구성하세요. 다만, Context 정보가 실제 사용자 질문과 연관이 없다면 무시하시고, 관련된 특허를 찾기 어렵지만 가지고 있는 지식 기반 답변을 제공해주세요.
"""

    # ✅ 메시지 전달
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt.format(context=context, question=user_query))
    ]

    # print("\n📦 [Context Preview - 요약 입력 내용]")
    # print(context[:5000])  # 너무 길 경우 앞 부분만

    return llm.invoke(messages).content.strip()


def selector_filter_context(context_groups: Dict[str, List[Document]], query: str, llm, log_list=[]):
    selected_docs = []
    selected_app_nums = []
    for app_num, docs in context_groups.items():
        example = docs[0]
        title = example.metadata.get("발명의 명칭", "")
        applicant = example.metadata.get("출원인", "")
        summary = "\n".join([
            f"[{doc.metadata.get('section', '')}]\n{doc.page_content[:700]}"
            for doc in docs if doc.metadata.get("section") != "독립청구항"
        ])

        prompt = f"""
        아래는 출원번호 {app_num}에 해당하는 특허입니다:
        - 발명의 명칭: {title}
        - 출원인: {applicant}
        - 주요 내용:
        {summary}

        사용자 질문:
        "{query}"

        위에서 뽑힌 특허의 내용이 사용자의 궁금증 및 질문을 해결하는데 기여할 수 있으면 'Y', 관련 없으면 'N'만 답하세요.
        """
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        decision = response[0].upper()
        log_list.append({"출원번호": app_num, "출원인": applicant, "발명의 명칭": title, "결정": decision})
        if decision == "Y":
            selected_docs.extend(docs)
            selected_app_nums.append(app_num)

    # ✅ 선택된 출원번호 출력
    # if selected_app_nums:
    #     print(f"\n✅ 선택된 출원번호 ({len(selected_app_nums)}개): {', '.join(selected_app_nums)}")
    # else:
    #     print("\n❌ Selector 판단 결과: 선택된 출원번호 없음")

    return selected_docs


def run_filtered_rag(vectorstore, embedding_fn, user_query, llm, use_bm25=True):
    #  필터링 조건 추론
    filtering_prompt = PromptTemplate.from_template("""
사용자의 질의:
"{query}"

사용자 쿼리에서 추출 가능한 정보는 다음과 같습니다.
- 출원인: 특허를 출원한 특정 기업 혹은 특정 사람의 이름을 의미함
ex) 현대자동차, 현대모비스, 엘지이노텍, 삼성에스디아이, 엘지전자, 테슬라, BYD 등
- 출원일자: 특허 출원 시점을 의미하며, 연도단위가 될 수 있음                                                                        

위 항목 중 어떤 항목을 어떤 조건으로 필터링해야 가장 적절한 검색이 될지 판단하되, 필터링 조건이 사용자의 질의에 없는데 굳이 할 필요는 없습니다.

사용자 질의에서 사용자가 요구하는 정보는 
1)출원인 
2)출원일자 
이며, 2가지의 정보 중 하나만 요구하고 있거나, 다 요구하고 있거나, 모두 요구하고 있지 않을 수 있으니, 이에 유의하여 filtering을 시행하세요.
이때, 반드시 **하나의 JSON 객체**만 작성하고 사용자 쿼리에서 특정 조건들이 추출되지 않을 경우는 **빈 JSON 형태를 추출하세요**
아래의 예시를 참고하여 추출하세요.

**출력 예시** (다른 문장은 쓰지 말고 이 형식만 출력하세요):
1. 예를 들어, "현대자동차가 출원하고, 2023년 이후에 출원된 구동모터 특허를 검색해줘" 라는 질의는 사용자가 출원인과 출원일자 기준 정보를 모두 요구하므로,
{{
    "출원인": [현대자동차],
    "출원일자": {{"$gte": 20230101}}
}}
형식이 가능합니다.                                                
2. 두 번째 예로,  "현대글로비스가 출원한 2022년 이전의 특허를 검색해줘" 라는 질의는 사용자가 출원인과 출원일자 기준 정보를 모두 요구하므로,
{{
    "출원인": [현대글로비스],
    "출원일자": {{"$lte": 20220101}}
}}
형식이 가능합니다.

3. 세 번째 예로, "현대모비스와 엘지이노텍이 출원한 2020년과 2023년 사이의 특허를 검색해줘" 라는 질의는 사용자가 2가지 출원인과 출원일자 정보를 모두 요구하므로,
{{
    "출원인": [현대모비스, 엘지이노텍],
    "출원일자":{{'$gte': 20220101, '$lte': 20240101}}
}}
형식이 가능합니다.                                                
4. 네 번째 예로, "2021년 이전의 전기차 배터리 관련 특허를 검색해줘" 라는 질의는 사용자가 출원일자 기준 정보만을 요구하므로,
{{
    "출원일자":{{'$lte': 20240101}}                                           
}}
형식이 가능합니다.                                                    
5. 다섯 번째 예로, "2023년에 나온 하이브리드 차 제동장치 관련 특허를 검색해줘" 라는 질의는 사용자가 출원일자 기준 정보를 요구하므로,
{{
    "출원일자":{{'$gte': 20230101 ,'$lte': 20240101}}                                           
}}
형식이 가능합니다.                                                    
6. 여섯번째 예로, "엘지이노텍의 전기차 배터리 관련 특허를 검색해줘" 라는 질의는 사용자가 출원인 기준 정보만을 요구하므로, 
{{
    "출원인":[엘지이노텍]                                          
}}
로 JSON 형식을 만들면 됩니다.

7. 마지막 예로, "전기차 배터리 관련 특허를 검색해줘" 라는 질의는 **출원인**과 **출원일자**에 대한 사용자의 정보 요구가 모두 존재하지 않으므로,
빈 형태의 JSON인
{{}}을 반환하면 됩니다.

참고로, 사용되는 형식에서 출원일자의 $gte 는 특정 시점 이후, $lte는 특정 시점 이전을 나타내는 표기입니다.
**특정 기업** 이라는 것으로 출원인 항목을 표기하지 마세요. 없는 경우는 JSON을 생성하지 않으면 됩니다.
현재 연도는 2025년입니다. 사용자의 '작년', '재작년', 'N년전' 등에 대한 자연어 질의에 초점을 맞추어 유동적으로 범위를 생성하세요.                                
"""
                                                    )
    filter_chain = LLMChain(llm=llm, prompt=filtering_prompt)
    try:
        parsed_filter = json.loads(extract_json_from_text(filter_chain.invoke(query=user_query)))
        filter_exists = True
    except:
        parsed_filter = {}
        filter_exists = False

    # print(f"🔍 추론된 필터 조건: {parsed_filter}")

    # 2. 전체 문서 준비
    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    # 3. Ensemble Retriever with weight
    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    all_docs = [Document(page_content=doc, metadata=meta) for doc, meta in
                zip(all_data["documents"], all_data["metadatas"])]

    dense_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})

    if use_bm25:
        all_data = vectorstore._collection.get(include=["documents", "metadatas"])
        all_docs = [Document(page_content=doc, metadata=meta) for doc, meta in
                    zip(all_data["documents"], all_data["metadatas"])]
        sparse_retriever = BM25Retriever.from_documents(all_docs)
        sparse_retriever.k = 10
        sparse_docs = sparse_retriever.invoke(user_query)
    else:
        sparse_docs = []  # 빈 리스트 처리

    dense_docs = dense_retriever.invoke(user_query)
    retrieved_docs = ensemble_with_normalized_scores(dense_docs, sparse_docs, top_k=10)
    sparse_retriever = BM25Retriever.from_documents(all_docs)
    sparse_retriever.k = 10

    dense_docs = dense_retriever.invoke(user_query)
    sparse_docs = sparse_retriever.invoke(user_query)
    retrieved_docs = ensemble_with_normalized_scores(dense_docs, sparse_docs, top_k=10)

    # print(f"\n🔎 총 retrieval 문서 수: {len(retrieved_docs)}개")
    # for i, doc in enumerate(retrieved_docs[:10]):
    #     print(f"[🔹 Retrieved {i+1}] {doc.metadata.get('출원번호', '')} | {doc.metadata.get('출원인', '')} | {doc.metadata.get('발명의 명칭', '')} | Section: {doc.metadata.get('section')}")

    if not retrieved_docs:
        # print("📭 관련 특허 문서를 찾지 못했습니다.")
        return "질문과 관련된 특허를 찾을 수 없습니다."

    # 4. 사후 필터링
    if filter_exists:
        filtered_docs = apply_post_filter(retrieved_docs, parsed_filter)
        if not filtered_docs:
            # print("📭 필터 후 문서 없음")
            filtered_docs = retrieved_docs
            top_docs = retrieved_docs[:3]
    else:
        filtered_docs = retrieved_docs
        top_docs = filtered_docs[:3]

    # print(f"\n✅ 필터링 적용 후 문서 수: {len(filtered_docs)}개")
    # for i, doc in enumerate(filtered_docs[:10]):
    #     print(f"[🔸 Filtered {i+1}] {doc.metadata.get('출원번호', '')} | {doc.metadata.get('출원인', '')} | {doc.metadata.get('발명의 명칭', '')} | Section: {doc.metadata.get('section')}")

    top_docs = filtered_docs[:3]

    # 5. 확장 (요약 대상, 구조화 정보 분리)
    summary_docs, structured_df = expand_documents_by_application_number(top_docs, vectorstore)
    summary_grouped = group_docs_by_application_number(summary_docs)

    # 6. Selector Loop (요약 대상, 구조화 정보 분리)
    selector_log = []
    selected_docs = selector_filter_context(summary_grouped, user_query, llm, selector_log)

    if not selected_docs:
        # print("🌀 Selector 전부 부적합 → Top 4~6 재시도")
        retry_docs = filtered_docs[3:6] if len(filtered_docs) >= 6 else retrieved_docs[3:6]
        retry_summary_docs, structured_df = expand_documents_by_application_number(retry_docs, vectorstore)
        retry_grouped = group_docs_by_application_number(retry_summary_docs)
        selected_docs = selector_filter_context(retry_grouped, user_query, llm, selector_log)

    #  다시 실패 시 3차 시도
    if not selected_docs:
        # print("🌀 Selector 재시도 실패 → Top 7~10 재시도")
        retry_docs = filtered_docs[6:10] if len(filtered_docs) >= 10 else retrieved_docs[6:10]
        retry_summary_docs, structured_df = expand_documents_by_application_number(retry_docs, vectorstore)
        retry_grouped = group_docs_by_application_number(retry_summary_docs)
        selected_docs = selector_filter_context(retry_grouped, user_query, llm, selector_log)

    if not selected_docs:
        # print("📭 최종적으로 적합한 특허 없음")
        # print("질문과 관련된 특허를 찾을 수 없습니다.")
        advanced_summary = "해당 기술과 관련한 특허를 찾기 어려움"
        structured_info_df = pd.DataFrame()
        return advanced_summary, structured_info_df

    # print(f"📚 Selector 통과 후 문서 개수: {len(selected_docs)}개")

    # 선택된 문서 기반 Summary 진행
    selected_app_nums = list(set(doc.metadata.get("출원번호", "") for doc in selected_docs))

    # 6. 고급 요약 생성
    advanced_summary = generate_advanced_summary(selected_docs, user_query, llm)
    summary_header = "\n📘 특허 검색 및 요약 정보:\n"
    full_summary = summary_header + advanced_summary

    # Selector 이후 출원번호 추출
    selected_app_nums = list({doc.metadata.get("출원번호") for doc in selected_docs})

    # Selector 통과 못한 출원번호 추출
    excluded_app_nums = list({
        doc.metadata.get("출원번호")
        for doc in retrieved_docs
        if doc.metadata.get("출원번호") not in selected_app_nums
    })

    # "요약" section만 vectorstore에서 재리트리브
    def retrieve_summary_sections_by_appnums(app_nums: List[str], vectorstore: Chroma, k: int = 20) -> List[Document]:
        """
        출원번호 리스트를 기준으로 각 출원번호별 문서를 리트리브하고,
        그 중 section이 '요약'인 문서만 필터링해서 반환한다.
        """
        retrieved = []
        for app_num in app_nums:
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "filter": {"출원번호": app_num},
                    "k": k
                }
            )
            docs = retriever.invoke("")
            retrieved.extend([doc for doc in docs if doc.metadata.get("section") == "요약"])
        return retrieved

    retrieved_excluded_summaries = retrieve_summary_sections_by_appnums(excluded_app_nums, vectorstore)

    # Selector 통과 못한 문서 요약 제공용 DF 생성

    st.markdown("## 📘 특허 검색 및 요약 정보")
    st.markdown(full_summary)

    excluded_summary_records = []
    for doc in retrieved_excluded_summaries:
        raw_summary = doc.page_content.strip()

        # 요약, 요약\n, 요약: 등으로 시작하는 부분 제거
        cleaned_summary = re.sub(r"^요약[\s:\n]*", "", raw_summary)

        excluded_summary_records.append({
            "출원번호": doc.metadata.get("출원번호", "N/A"),
            "출원인": doc.metadata.get("출원인", "N/A"),
            "발명의 명칭": doc.metadata.get("발명의 명칭", "N/A"),
            "요약": cleaned_summary
        })

    excluded_summary_df = pd.DataFrame(excluded_summary_records)

    if not excluded_summary_df.empty:
        st.markdown("### 📄 관련 특허 요약 정보")
        st.dataframe(
            excluded_summary_df[["출원번호", "발명의 명칭", "출원인", "요약"]]
        )
    return advanced_summary, excluded_summary_df
