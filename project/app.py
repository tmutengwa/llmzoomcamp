import os
import time
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from search import FastAPIQAAssistant

# Set page config
st.set_page_config(
    page_title="FastAPI QA Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- DB Initialization & Logging -----------------
def init_logging_db():
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            search_type TEXT,
            enable_rewriting BOOLEAN,
            enable_reranking BOOLEAN,
            answer TEXT,
            response_time_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            rating INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(query_id) REFERENCES queries(query_id)
        )
    """)
    conn.commit()
    conn.close()

def log_query(query_text, search_type, enable_rewriting, enable_reranking, answer, response_time_ms, input_tokens, output_tokens):
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO queries (query_text, search_type, enable_rewriting, enable_reranking, answer, response_time_ms, input_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (query_text, search_type, enable_rewriting, enable_reranking, answer, response_time_ms, input_tokens, output_tokens))
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return query_id

def log_feedback(query_id, rating, comment=""):
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (query_id, rating, comment)
        VALUES (?, ?, ?)
    """, (query_id, rating, comment))
    conn.commit()
    conn.close()

# Initialize DB on start
init_logging_db()

# Initialize assistant
@st.cache_resource
def get_assistant():
    return FastAPIQAAssistant()

try:
    assistant = get_assistant()
except Exception as e:
    st.error(f"Error loading indexes: {e}. Did you run the ingestion pipeline `ingest.py`?")
    st.stop()

# ----------------- UI / App Navigation -----------------
st.sidebar.title("⚡ FastAPI QA Assistant")
page = st.sidebar.radio("Navigation", ["Chat Assistant", "Monitoring Dashboard"])

if page == "Chat Assistant":
    st.title("⚡ FastAPI QA Chat Assistant")
    st.write("Ask technical questions about FastAPI features, routers, dependency injection, and security.")
    
    # Sidebar options
    st.sidebar.subheader("RAG Parameters")
    search_type = st.sidebar.selectbox("Search Mode", ["hybrid", "vector", "text"], help="Select lexical (text), semantic (vector), or combined (hybrid RRF) search.")
    enable_rewriting = st.sidebar.checkbox("Query Rewriting", value=False, help="LLM rewrites and expands user query for broader context coverage.")
    enable_reranking = st.sidebar.checkbox("Re-rank Results", value=False, help="LLM evaluates and sorts retrieved documents before context injection.")
    
    # Initialize message list
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Retrieved Context Documents"):
                    for src in msg["sources"]:
                        st.markdown(f"**Source File:** `{src['filename']}`")
                        st.code(src["content"][:400] + "...")
                        
            # Render feedback if query_id is logged
            if msg["role"] == "assistant" and "query_id" in msg:
                q_id = msg["query_id"]
                # Check if feedback already submitted in session
                feedback_key = f"feedback_submitted_{q_id}"
                if feedback_key not in st.session_state:
                    col1, col2, _ = st.columns([0.1, 0.1, 0.8])
                    with col1:
                        if st.button("👍", key=f"up_{q_id}"):
                            log_feedback(q_id, 1)
                            st.session_state[feedback_key] = True
                            st.success("Feedback saved!")
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"down_{q_id}"):
                            log_feedback(q_id, -1)
                            st.session_state[feedback_key] = True
                            st.error("Feedback saved!")
                            st.rerun()
                else:
                    st.caption("Thank you for your feedback!")

    # Chat Input
    if user_query := st.chat_input("Ask a question (e.g., 'How do I do dependency injection?')"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving documentation and generating answer..."):
                start_time = time.time()
                
                # Retrieve context
                context_docs = assistant.retrieve(
                    query=user_query,
                    search_type=search_type,
                    enable_rewriting=enable_rewriting,
                    enable_reranking=enable_reranking,
                    num_results=5
                )
                
                # Generate final RAG answer
                answer, usage = assistant.generate_answer(user_query, context_docs)
                
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Log to DB
                query_id = log_query(
                    query_text=user_query,
                    search_type=search_type,
                    enable_rewriting=enable_rewriting,
                    enable_reranking=enable_reranking,
                    answer=answer,
                    response_time_ms=response_time_ms,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"]
                )
                
            st.markdown(answer)
            with st.expander("Retrieved Context Documents"):
                for src in context_docs:
                    st.markdown(f"**Source File:** `{src['filename']}`")
                    st.code(src["content"][:400] + "...")

        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": context_docs,
            "query_id": query_id
        })
        st.rerun()

elif page == "Monitoring Dashboard":
    st.title("📊 In-App Monitoring Dashboard")
    st.write("Live charts and user feedback stats from queries logged in `logs.db`.")
    
    # Read stats from sqlite
    conn = sqlite3.connect("logs.db")
    
    df_queries = pd.read_sql_query("SELECT * FROM queries", conn)
    df_feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    
    if df_queries.empty:
        st.info("No queries logged yet. Chat with the assistant first!")
        st.stop()
        
    # Process Timestamps
    df_queries["timestamp"] = pd.to_datetime(df_queries["timestamp"])
    
    # ---------------- Metrics Row ----------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", len(df_queries))
    with col2:
        avg_rt = df_queries["response_time_ms"].mean()
        st.metric("Avg Response Time", f"{avg_rt:.0f} ms" if not pd.isna(avg_rt) else "N/A")
    with col3:
        total_tokens = df_queries["input_tokens"].sum() + df_queries["output_tokens"].sum()
        st.metric("Total Token Cost (tokens)", f"{total_tokens:,}")
    with col4:
        pos_feed = len(df_feedback[df_feedback["rating"] == 1])
        neg_feed = len(df_feedback[df_feedback["rating"] == -1])
        st.metric("Thumbs Up / Down Ratio", f"{pos_feed} / {neg_feed}")

    st.markdown("---")
    
    # ---------------- Charts ----------------
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📈 Chart 1: Daily Query Volume")
        df_queries["date"] = df_queries["timestamp"].dt.date
        df_daily = df_queries.groupby("date").size().reset_index(name="Count")
        fig_daily = px.line(df_daily, x="date", y="Count", labels={"date": "Date", "Count": "Queries"}, template="plotly_dark")
        st.plotly_chart(fig_daily, use_container_width=True)
        
    with row1_col2:
        st.subheader("😊 Chart 2: User Feedback Distribution")
        if not df_feedback.empty:
            df_feed_counts = df_feedback.groupby("rating").size().reset_index(name="Count")
            df_feed_counts["Rating Type"] = df_feed_counts["rating"].map({1: "Positive 👍", -1: "Negative 👎"})
            fig_feed = px.pie(df_feed_counts, values="Count", names="Rating Type", color="Rating Type",
                             color_discrete_map={"Positive 👍": "#4CAF50", "Negative 👎": "#F44336"},
                             template="plotly_dark")
            st.plotly_chart(fig_feed, use_container_width=True)
        else:
            st.info("No feedback submitted yet.")

    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.subheader("⚡ Chart 3: Response Time Distribution")
        fig_rt = px.histogram(df_queries, x="response_time_ms", nbins=20,
                             labels={"response_time_ms": "Response Time (ms)"},
                             color_discrete_sequence=["#FF9800"], template="plotly_dark")
        st.plotly_chart(fig_rt, use_container_width=True)

    with row2_col2:
        st.subheader("🤖 Chart 4: Token Consumption over Time")
        df_daily_tokens = df_queries.groupby("date")[["input_tokens", "output_tokens"]].sum().reset_index()
        fig_tokens = px.bar(df_daily_tokens, x="date", y=["input_tokens", "output_tokens"],
                            labels={"value": "Tokens", "date": "Date", "variable": "Token Type"},
                            template="plotly_dark")
        st.plotly_chart(fig_tokens, use_container_width=True)

    st.subheader("🔍 Chart 5: Search Mode Popularity")
    df_mode = df_queries.groupby("search_type").size().reset_index(name="Count")
    fig_mode = px.bar(df_mode, x="search_type", y="Count", labels={"search_type": "Search Mode", "Count": "Usage Count"},
                      color="search_type", template="plotly_dark")
    st.plotly_chart(fig_mode, use_container_width=True)

    # Detailed Logs Table
    st.markdown("### 📋 Recent Queries Log")
    st.dataframe(df_queries[["query_id", "query_text", "search_type", "response_time_ms", "timestamp"]].sort_values("timestamp", ascending=False).head(20))
