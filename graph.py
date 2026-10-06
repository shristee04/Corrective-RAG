import os
import json
from dotenv import load_dotenv

from tavily import TavilyClient
from typing import TypedDict, List
from langchain_groq import ChatGroq
from ingest import load_vectorstore
load_dotenv()

llm= ChatGroq( model="openai/gpt-oss-20b", temperature=0)
tavily= TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

max_retries= 2
# Shape of the data that flows through every node in the graph 
class Ragstate(TypedDict):
    question:str
    sub_questions: List[str]
    documents: List[str]
    source: str
    answer: str
    retries: int
    chat_history: List[str]

# break the question into sub question
def decompose_question(state: Ragstate)-> Ragstate:
    prompt = f"""Break the following question into 1-3 simple, specific sub-questions
that together would fully answer it. If it's already simple, just return it as-is.
Respond ONLY as a JSON list of strings, nothing else.

Question: {state['question']}"""
    
    response= llm.invoke(prompt).content
    try:
        sub_questions = json.loads(response)

    except json.JSONDecodeError:
        # Fallback: if the llm didn't return valid json , just use the original question
        sub_questions= [state['question']]

    return {**state, "sub_questions": sub_questions}

# Searches the FAISS index for chunks matching each sub questions
def retrieve(state: Ragstate)-> Ragstate:
    vectorstore= load_vectorstore()
    retriever= vectorstore.as_retriever(search_kwargs={"k": 4})

    all_docs= []
    for q in state["sub_questions"]:
        results= retriever.invoke(q)
        all_docs.extend([doc.page_content for doc in results] )

    return {**state, "documents": all_docs, "source": "vectorstore"}

# Check what we got is relevant or not to a question.
def grade_relevance(state: Ragstate)-> str:
    context= "\n\n".join(state["documents"])[:3000]
    prompt = f"""You are grading whether retrieved context is relevant to a question.
    Questions: {state['question']}
    Context: {context}

Is this context relevant enough to answer the question? Reply with ONLY "yes" or "no". """

    verdict= llm.invoke(prompt).content.strip().lower()
    return "generate" if "yes" in verdict else "web_search"

# backup plan: if the pdf doesn't have the relevant info just web search 
def web_search_fallback(state: Ragstate)-> Ragstate:
    results= tavily.search(query=state["question"], max_results= 4)
    web_docs= [r["content"] for r in results.get("results", [])]
    return {**state, "documents": web_docs, "source": "web"}

#generate the actuall answer using the matching chunks to our quetsion if it is relevant
def generate_answer(state: Ragstate)-> Ragstate:
    context= "\n\n".join(state["documents"])[:4000]
    prompt= f"""Answer the question using ONLY the context below. If the context
doesn't contain the answer, say so honestly - do not make things up.

Context: {context}
Questions: {state["question"]}
answer:
"""
    answer= llm.invoke(prompt).content
    return {**state, "answer": answer}

# Double check the answer that it grounded in the context and actually on topic
def verify_answer(state: Ragstate)-> Ragstate:
    prompt= f"""Check this answer against two criteria.
1. GROUNDED: Is the answer actually supported by the context (not made up)?
2. RELEVANT: Does the answer actually address the original question?

context: {' '.join(state["documents"])[:3000]}
Questions: {state["question"]}
Answer: {state["answer"]}

Reply ONLY as JSON: {{"grounded": true/false, "relevant": true/false}}
"""
    response= llm.invoke(prompt).content
    try:
        verdict= json.loads(response)
    # if the llm check itself fails to parse, don't get stuck in a loop - just pass
    except json.JSONDecodeError:
        verdict= {"grounded": True, "relevant": True}

    passed= verdict.get("grounded") and verdict.get("relevant")
    if passed or state["retries"]> max_retries:
        return "end"
    return "retry"

# bumps the retry counter before we loop back and try generating again
def increment_retry(state: Ragstate)-> Ragstate:
    return {**state, "retries": state["retries"]+1}

# creating chat history
def rewrite_question(state: Ragstate)-> Ragstate:
    if not state['chat_history']:
        return {**state, "question": state["question"]}
    
    prompt= f"""Given this convertation chat history and follow up question, 
    rewrite the foolow up as standalone question that include all necessary context
    History: {state['chat_history']}
Follow-up question: {state['question']}
Standalone question: """

    rewritten_question = llm.invoke(prompt).content
    return {**state, "question": rewritten_question}

#wires all the nodes together into an actuall CRAG pipeline
def build_graph():
    from langgraph.graph import StateGraph, END

    workflow= StateGraph(Ragstate)

    workflow.add_node("decompose", decompose_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search_fallback)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("rewrite", rewrite_question)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "decompose")
    workflow.add_edge("decompose", "retrieve")

# Branch: go straight to answering or fall back to web search first
    workflow.add_conditional_edges(
            "retrieve",
            grade_relevance,
            {"generate": "generate", "web_search": "web_search"},
        )
    workflow.add_edge("web_search", "generate")

# branch: end if the answer is verified else go back otherwise loop back and retry
    workflow.add_conditional_edges(
        "generate",
        verify_answer,
        {"end": END, "retry": "increment_retry" }
    )

    workflow.add_edge("increment_retry", "retrieve")

    return workflow.compile()

# entry point called by app.py - runs one question through the whole pipeline
def ask(question: str, chat_history: list = None) -> dict:
    if chat_history is None:
        chat_history = []
    
    graph = build_graph()
    initial_state: Ragstate = {
        "question": question,
        "sub_questions": [],
        "documents": [],
        "source": "",
        "answer": "",
        "retries": 0,
        "chat_history": chat_history
    }
    return graph.invoke(initial_state)

graph = build_graph()

# Save the graph as a PNG image
graph.get_graph().draw_mermaid_png(output_file_path="graph_diagram.png")


