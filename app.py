import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from agent import build_agent_graph, AgentState
from utils import load_excel

load_dotenv()

st.set_page_config(page_title="Excel Data Agent", layout="wide")
st.title("📊 LLM-Powered Excel Data Agent")

st.markdown("Upload an Excel file and ask questions about your data!")

graph = build_agent_graph()

if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState()

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = load_excel(uploaded_file)
    st.session_state.agent_state["df"] = df
    st.write("✅ File uploaded successfully!")
    st.dataframe(df)

    if st.button("Process Data"):
        st.session_state.agent_state = graph.invoke(st.session_state.agent_state)
        st.success("Data processed and vectorized!")

if st.session_state.agent_state.get("vectorstore", None):
    query = st.text_input("Ask a question about your data")

    if query:
        st.session_state.agent_state["question"] = query
        st.session_state.agent_state = graph.invoke(st.session_state.agent_state)
        st.write("💡 Answer:")
        st.success(st.session_state.agent_state["answer"])
