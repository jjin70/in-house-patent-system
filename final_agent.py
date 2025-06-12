from typing import Union, List, Tuple, Dict
from pydantic import BaseModel
import json
import pandas as pd
import re
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from rag_tool_memory_save_noprint import run_filtered_rag
from Final_Trend import KeywordAnalyzer
from Final_evaluator import Agent3
from writer_tool4_new import generate_technical_draft

llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)

def safe_result_summary(result):
    if isinstance(result, pd.DataFrame):
        return result.to_markdown(index=False)
    elif isinstance(result, list):
        return "\n".join(str(x) for x in result[:5])
    elif isinstance(result, dict):
        return json.dumps(result, indent=2, ensure_ascii=False)
    else:
        return str(result)

def extract_json_from_text(text: str) -> str:
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text)
    if match:
        return match.group(0)
    raise ValueError("JSON 형식이 아닙니다.")

embedding_model = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(
    persist_directory="/Users/heejinyang/python/streamlit/chroma_db_streamlit",
    embedding_function=embedding_model,
)

class PlanExecute(BaseModel):
    input: str
    tools: List[str] = []
    sub_queries: Dict[str, Union[str, List[str]]] = {}
    results: Dict[str, str] = {}
    response: Union[str, None] = None
    log: List[str] = []
    llm: ChatOllama = llm  # 전역 공유 LLM

tool_selector_prompt = ChatPromptTemplate.from_template("""
다음 사용자의 질문을 해결하기 위해 어떤 도구를 사용하는 것이 가장 적절한지 판단하세요.

[도구 설명]
- patent_searcher: 사용자의 기술적 질문에 맞는 특허를 검색하고 그 내용을 요약합니다.
- patent_trend_analyzer: 특정 키워드, 출원인을 기준으로 특허 출원 건수 및 트렌드를 시각화합니다.
- patent_evaluator: 주어진 기술 주제에 대한 특허들을 평가하기 위해 자사가 중요하게 보는 지표를 토대로 경쟁사의 중요 관련 특허들을 분석합니다.
- tech_writer: 기술 개요를 바탕으로 기술 설명서 초안을 작성합니다.

사용자의 질문:
"{query}"

반드시 아래와 같은 형태의 JSON 형식으로 응답하세요:
{{"tools": ["patent_searcher"]}}
또는
{{"tools": ["tech_writer"]}}
""")

tool_selector_chain = tool_selector_prompt | llm

def tool_selector(state: PlanExecute):
    response = tool_selector_chain.invoke({"query": state.input})
    raw = response.content.strip()
    print("🛠 도구 선택 응답 원본:\n", raw)
    try:
        json_text = extract_json_from_text(raw)
        parsed = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"[tool_selector] JSON 파싱 실패: {e}\n원본 응답:\n{raw}")
    print(f"🛠 선택된 도구: {parsed['tools']}")
    return {"tools": parsed["tools"], "log": state.log + [f"🔧 선택된 도구: {parsed['tools']}"]}

sub_query_prompt = ChatPromptTemplate.from_template(r"""
사용자의 질문:
"{query}"

선택된 도구 목록:
{tools}

⚠️ 출력 지침 (중요):
- 반드시 중괄호 {{}}를 포함한 완전한 JSON 객체 형태로 출력하세요.
- 예시:
```json
{{
  "sub_queries": {{
    "patent_searcher": "전기차 배터리 효율 개선 기술",
    "patent_trend_analyzer": "2020년 이후 전기차 배터리 동향"
  }}
}}
""")

sub_query_chain = sub_query_prompt | llm

def tool_query_planner(state: PlanExecute):
    result = sub_query_chain.invoke({"query": state.input, "tools": ", ".join(state.tools)})
    print("🧩 Sub-query 응답 원본:\n", result.content)
    try:
        json_text = extract_json_from_text(result.content)
        parsed = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"❌ Sub-query 생성 결과 파싱 실패:\n{result.content}\n\n에러: {e}")
    print("🧩 Sub-query 분리 결과:", parsed["sub_queries"])
    return {"sub_queries": parsed["sub_queries"], "log": state.log + [f"🧩 Sub-query 분리 결과: {parsed['sub_queries']}"]}

def execute_tools(state: PlanExecute):
    results = {}
    for tool_name, query in state.sub_queries.items():
        if tool_name == "patent_searcher":
            sub_queries = [q.strip() for q in query.split(",") if q.strip()]
            combined_summary = ""
            for i, sub_q in enumerate(sub_queries, 1):
                summary, _ = run_filtered_rag(vectorstore, embedding_model, sub_q, llm=state.llm, use_bm25=True)
                combined_summary += f"[서브 쿼리 {i}] {sub_q}\n{summary}\n\n"
            results[tool_name] = combined_summary.strip()
        elif tool_name == "patent_trend_analyzer":
            agent = KeywordAnalyzer(csv_path="/Users/heejinyang/python/streamlit/Codes/0527_cleaning_processing_ver1.csv", llm=state.llm)
            interpretation = agent.run(query)
            results[tool_name] = safe_result_summary(interpretation)
        elif tool_name == "patent_evaluator":
            agent = Agent3(csv_path="/Users/heejinyang/python/streamlit/Codes/0527_cleaning_processing_ver1.csv", llm=state.llm)
            interpretation = agent.handle(topic_query=query)
            results[tool_name] = interpretation
        else:
            results[tool_name] = f"[{tool_name}]에 대한 응답 (Mock 처리됨)"
    return {"results": results, "log": state.log + [f"⚙️ 실행 완료: {list(results.keys())}"]}

def post_summary(state: PlanExecute):
    merged = "\n\n".join(f"[{tool} 결과]\n{res}" for tool, res in state.results.items())
    if not merged.strip():
        return {"response": "❗ 도구 실행 결과가 비어 있습니다.", "log": state.log + ["⚠️ post_summary: 결과 없음"]}
    summary_prompt = PromptTemplate.from_template("""
아래는 각 도구를 통해 수집된 결과입니다:

{merged}

이 정보를 바탕으로 사용자의 질문에 대한 종합적인 답변을 자연스럽게 작성하세요.
이때, 각 Tool에 따라 어떤 정보가 파악되었는지, 이를 통해 어떤 종합적 시사점을 얻을 수 있는지의 형식으로 출력할 수 있습니다.
- 이때, **반드시** 한국어로만 결과를 출력하세요.
""")
    summary_chain = summary_prompt | state.llm
    result = summary_chain.invoke({"merged": merged})

    print("🧠 최종 응답 요약:\n", result.content.strip())
    return {"response": result.content.strip(), "log": state.log + ["🧠 최종 종합 요약 완료"]}

graph = StateGraph(PlanExecute)
graph.add_node("tool_selector", tool_selector)
graph.add_node("tool_query_planner", tool_query_planner)
graph.add_node("execute", execute_tools)
graph.add_node("post_summary", post_summary)

graph.add_edge(START, "tool_selector")
graph.add_edge("tool_selector", "tool_query_planner")
graph.add_edge("tool_query_planner", "execute")
graph.add_edge("execute", "post_summary")
graph.add_edge("post_summary", END)

app = graph.compile(checkpointer=MemorySaver())
