from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from langchain.document_loaders import DataFrameLoader
import pandas as pd
import os

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# Global to store index
vectorstore = None

def ingest_dataframe(df: pd.DataFrame):
    loader = DataFrameLoader(df)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    return FAISS.from_documents(splits, embeddings)

def create_qa_chain(vstore):
    retriever = vstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# LangGraph state
class AgentState(dict):
    df: pd.DataFrame = None
    vectorstore: FAISS = None
    question: str = ""
    answer: str = ""

def upload_node(state: AgentState) -> AgentState:
    state["vectorstore"] = ingest_dataframe(state["df"])
    return state

def query_node(state: AgentState) -> AgentState:
    qa_chain = create_qa_chain(state["vectorstore"])
    result = qa_chain.run(state["question"])
    state["answer"] = result
    return state

def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("ingest", upload_node)
    graph.add_node("ask", query_node)
    
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "ask")
    graph.add_edge("ask", END)
    
    return graph.compile()
