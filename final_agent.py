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

# ✅ Embedding + Vectorstore 초기화
embedding_model = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma(
    persist_directory="/Users/heejinyang/python/streamlit/chroma_db_streamlit",
    embedding_function=embedding_model,
)

# ✅ 상태 정의
class PlanExecute(BaseModel):
    input: str
    tools: List[str] = []
    sub_queries: Dict[str, Union[str, List[str]]] = {}
    sub_queries_generated: bool = False  # ✅ 여기에 추가!
    results: Dict[str, str] = {}
    response: Union[str, None] = None
    log: List[str] = []
    llm: ChatOllama = llm # 전역 공유 llm

# ✅ Tool Selector (도구 설명 포함)
tool_selector_prompt = ChatPromptTemplate.from_template("""
다음 사용자의 질문을 해결하기 위해 어떤 도구를 사용하는 것이 가장 적절한지 판단하세요.

[도구 설명]
- patent_searcher: 사용자의 기술적 질문에 맞는 특허를 검색하고 그 내용을 요약합니다.
- patent_trend_analyzer: 특정 키워드, 출원인을 기준으로 특허 출원 건수 및 트렌드를 시각화합니다.
- patent_evaluator: 주어진 기술 주제에 대한 특허들을 평가하기 위해 자사가 중요하게 보는 지표를 토대로 경쟁사의 중요 관련 특허들을 분석합니다.
- tech_writer: 기술 개요를 바탕으로 기술 설명서 초안을 작성합니다.

사용자의 질문:
"{query}"

✅ 선택 지침:
- 하나의 도구만 선택하는 것이 원칙입니다.
- 단, 질문이 두 개 이상의 도구가 협업해야만 해결되는 경우, 도구를 복수 선택하세요.
- patent_searcher의 경우 질문이 두 가지 이상의 고도화된 특허 검색이 요구되는 경우, patent_searcher를 두 번 반복 호출해도 됩니다.
- patent_searcher가 2번 선택된 경우는 **그리고**, **또는** 혹은 마침표나 콤마로 서로 다른 성격의 기술에 대해서 질문하는 경우에 두 번 호출하면돼.

⚠️ 반드시 유효한 JSON 형식으로 출력하세요. 마지막 줄은 "}}" 로 정확히 닫아야 하며, 쉼표 누락이 없어야 합니다.
                                                                                                                
반드시 아래와 같은 형태의 JSON 형식으로 응답하세요:
{{ "tools": ["patent_searcher"] }}
또는
{{ "tools": ["patent_searcher", "patent_evaluator"] }}                                                   
또는 사용자가 두 가지 상이한 기술적 내용 분석을 요구하는 경우,                                                        
{{ "tools": ["patent_searcher", "patent_searcher"] }}                                                        
또는
{{ "tools": ["patent_searcher", "patent_trend_analyzer"] }}
또는
{{ "tools": ["patent_searcher", "patent_trend_analyzer","patent_evaluator"]}}   
또는
{{ "tools": ["tech_writer"] }}                                                                                                                                                                       
""")
tool_selector_chain = tool_selector_prompt | llm

def tool_selector(state: PlanExecute):
    # 1) LLM 응답을 그대로 받고
    response = tool_selector_chain.invoke({"query": state.input})
    raw = response.content.strip()

    try:
        # 2) extract_json_from_text 로 JSON 덩어리만 추출
        json_text = extract_json_from_text(raw)
        parsed = json.loads(json_text)
    except Exception as e:
        # 3) 실패 시 디버깅을 위해 원본 응답 남기기
        raise ValueError(
            f"[tool_selector] JSON 파싱 실패: {e}\n"
            f"원본 응답:\n{raw}"
        )

    # 4) 잘 파싱된 tools 리스트로 상태 갱신
    return {
        "tools": parsed["tools"],
        "log": state.log + [f"🔧 선택된 도구: {parsed['tools']}"]
    }

# ✅ Sub-query Planner
sub_query_prompt = ChatPromptTemplate.from_template("""
다음은 사용자 질문과 선택된 도구 목록입니다.
각 도구에 대해 하나의 질의를 생성하세요. 
단,
- patent_searcher는 질문을 반영하여 기술적인 질문을 생성하여 전달, 이때 사용자가 기술적인 내용을 담기에 내용을 임의대로 누락하지 않았으면 해. **그리고**, **또는** 및 마침표 콤마로 구분해서 두 가지 기술에 대해서 질문을 하는 경우는 나누어서 파악해.
- patent_trend_analyzer는 '**특허 출원 동향 분석**'을 위해 사용자 sub query에 있는 키워드 단어와 연도를 사용해서 전달 (예1: 구동모터 관련 2020년 이후 출원 동향. 예2: 현대자동차의 배터리 관련 연도별 출원동향.)
- patent_evaluator는 경쟁사 및 특정 기술에 대한 평가를 위해 분석을 위한 키워드를 1~2개 사용해서 전달. 이때 키워드는 기술 단위여야 하고 자연어 형태로 전달해야 함.
- tech_writer는 "초안작성" 이라고 전달.

사용자의 질문:
"{query}"

선택된 도구 목록:
{tools}

복합 질의의 경우 출력 예시:

1) 사용자의 쿼리가 "전기차의 배터리 효율 개선 기술 관련 특허있어? 관련 기술에 대한 연도별 출원 동향도 알려줘"라면,  
{{ '{{' }}
  "sub_queries": {{ '{' }}
    "patent_searcher": "전기차의 배터리 효율 개선 기술 관련 특허",
    "patent_trend_analyzer": "전기차 과열 방지 관련 연도별 출원 동향"
  {{ '}' }}
{{ '}}' }}

2) 사용자의 쿼리가 "전기차 내 냉각수 흐름을 제어하기 위해 구축한 자동 모니터링 기술을 알고 싶고 열을 관리하기 위해 내부 쿨링시스템을 이용하는 기술을 알고 싶다" 고 한다면,  
{{ '{{' }}
  "sub_queries": {{ '{' }}
    "patent_searcher": "전기차 내 냉각수 흐름을 제어하기 위해 구축한 자동 모니터링하는 기술, 열 관리를 위한 차체 내 쿨링 시스템을 이용하는 기술"
  {{ '}' }}
{{ '}}' }}

3) 사용자의 쿼리가 "전기차의 변속감을 좋게 하기 위해 등장한 특허 있어? 관련 기술을 출원한 현대자동차의 특허 평가를 해줘"라면,  
{{ '{{' }}
  "sub_queries": {{ '{' }}
    "patent_searcher": "변속감을 주며 승차감을 높인 기술",
    "patent_evaluator": "변속감 및 속도 제어"
  {{ '}' }}
{{ '}}' }}
""", template_format="jinja2")

sub_query_chain = sub_query_prompt | llm

def tool_query_planner(state: PlanExecute):
    # ✅ 이미 분리되었으면 다시 실행하지 않음
    if state.sub_queries_generated:
        return {
            "sub_queries": state.sub_queries,
            "log": state.log + ["✅ Sub-query 이미 생성됨, 재생성하지 않음"]
        }

    tools_str = ", ".join(state.tools)
    result = sub_query_chain.invoke({"query": state.input, "tools": tools_str})

    try:
        json_text = extract_json_from_text(result.content)
        parsed = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"❌ Sub-query 생성 결과 파싱 실패:\n{result.content}\n\n에러: {e}")

    return {
        "sub_queries": parsed["sub_queries"],
        "sub_queries_generated": True,  # ✅ 상태값 갱신
        "log": state.log + [f"🧩 Sub-query 분리 결과: {parsed['sub_queries']}"]
    }

# ✅ Tool 실행기
def execute_tools(state: PlanExecute):
    results = {}
    for tool_name, query in state.sub_queries.items():
        if tool_name == "patent_searcher":
            sub_queries = [q.strip() for q in query.split(",") if q.strip()]  # 쉼표 기준 분리
            combined_summary = ""
            for i, sub_q in enumerate(sub_queries, 1):
                # print(f"\n🔍 [Tool1-{i}] 서브 쿼리 실행 중: {sub_q}")
                summary, _ = run_filtered_rag(vectorstore, embedding_model, sub_q, llm=state.llm, use_bm25=True)
                combined_summary += f"[서브 쿼리 {i}] {sub_q}\n{summary}\n\n"
            results[tool_name] = combined_summary.strip()


        elif tool_name == "patent_trend_analyzer":
            trend_agent = KeywordAnalyzer(csv_path="/Users/heejinyang/python/streamlit/Codes/0527_cleaning_processing_ver1.csv", llm=state.llm)
            intepretation = trend_agent.run(query) 
            results[tool_name] = safe_result_summary(intepretation)
        elif tool_name == "patent_evaluator":
            evaluator = Agent3(
                csv_path="/Users/heejinyang/python/streamlit/Codes/0527_cleaning_processing_ver1.csv",
                llm=state.llm
            )
            interpretation = evaluator.handle(topic_query=query)
            results[tool_name] = interpretation  # 바로 시사점 결과 저장
        elif tool_name == "tech_writer":
            print("📝 기술 설명서를 위한 초안 작성을 시작합니다.")
            result = generate_technical_draft.invoke({"user_input": query})
    
            # LangChain LLM 응답 객체일 경우 .content 추출
            content = result.content if hasattr(result, "content") else str(result)
            results[tool_name] = content        
        else:
            results[tool_name] = f"[{tool_name}]에 대한 응답 (Mock 처리됨)"
            
    return {
        "results": results,
        "log": state.log + [f"⚙️ 실행 완료: {list(results.keys())}"]
    }

# ✅ 결과 요약기
def post_summary(state: PlanExecute):
    merged = "\n\n".join(f"[{tool} 결과]\n{res}" for tool, res in state.results.items())
    
    if not merged.strip():
        return {
            "response": "❗ 도구 실행 결과가 비어 있습니다. 입력을 확인해 주세요.",
            "log": state.log + ["⚠️ post_summary: 결과 없음"]
        }
        
    summary_prompt = PromptTemplate.from_template("""
아래는 각 도구를 통해 수집된 결과입니다:

{merged}

이 정보를 바탕으로 사용자의 질문에 대한 종합적인 답변을 자연스럽게 작성하세요.
이때, 각 Tool에 따라 어떤 정보가 파악되었는지, 이를 통해 어떤 종합적 시사점을 얻을 수 있는지의 형식으로 출력할 수 있습니다.
- 단,다른 언어라면 **반드시** 한국어로 변환해서 출력하세요.
""")
    summary_chain = summary_prompt | state.llm
    result = summary_chain.invoke({"merged": merged})
    return {
        "response": result.content.strip(),
        "log": state.log + ["🧠 최종 종합 요약 완료"]
    }

# ✅ LangGraph 구성
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
