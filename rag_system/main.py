import streamlit as st
import pandas as pd
import plotly.express as px
import time
from database.neo4j_setup import test_connection
from retrieval.answer_generation import process_user_query, generate_final_answer
from retrieval.cypher_generation import generate_cypher_query, execute_cypher_query
from ingest.data_ingestion import ingest_all_data, load_embedding_model
from config import GROQ_API_KEY

# Set page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .dashboard-metric {
        font-size: 2rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .dashboard-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .success-text {
        color: #059669;
    }
    .warning-text {
        color: #D97706;
    }
    .error-text {
        color: #DC2626;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'neo4j_connected' not in st.session_state:
    st.session_state.neo4j_connected = test_connection()

if 'embedding_model' not in st.session_state:
    try:
        st.session_state.embedding_model = load_embedding_model()
        st.session_state.embedding_model_loaded = True
    except Exception as e:
        st.session_state.embedding_model_loaded = False
        st.session_state.embedding_model_error = str(e)

# Helper functions
def get_database_statistics():
    """Get Neo4j database statistics"""
    try:
        # This function would be implemented in database/neo4j_setup.py
        stats = {
            'nodes': {},
            'relationships': {},
            'top_authors': {},
            'top_topics': {},
            'model_performance': []
        }
        return stats
    except Exception as e:
        st.error(f"Error getting database statistics: {str(e)}")
        return {
            'nodes': {},
            'relationships': {},
            'top_authors': {},
            'top_topics': {},
            'model_performance': []
        }

def process_query_results(results, query_type):
    """Process and format query results for visualization"""
    if not results:
        return None
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Return early if DataFrame is empty or has unsuitable data
    if df.empty or len(df.columns) < 2:
        return None
    
    return df

def render_dashboard(df, query_type):
    """Render appropriate dashboard visualizations based on data"""
    if df is None or df.empty:
        st.info("No suitable data for visualization")
        return
    
    # Make sure numeric columns are actually numeric
    for col in df.columns:
        if col not in ['name', 'model', 'metric', 'author', 'topic', 'date', 'year']:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    # For model comparison
    if 'model' in df.columns and any(col for col in df.columns if col not in ['model', 'metric']):
        numeric_cols = [col for col in df.columns if col not in ['model', 'metric']]
        if numeric_cols:
            st.subheader("Model Performance Comparison")
            fig = px.bar(
                df, 
                x='model', 
                y=numeric_cols[0],
                color='model',
                title=f"Model Performance by {numeric_cols[0].capitalize()}",
                labels={'model': 'Model', numeric_cols[0]: numeric_cols[0].capitalize()}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Data Table")
            st.dataframe(df, use_container_width=True)
    
    # For author/topic analysis
    elif any(col in df.columns for col in ['author', 'topic']) and 'count' in df.columns:
        entity_col = 'author' if 'author' in df.columns else 'topic'
        st.subheader(f"{entity_col.capitalize()} Contribution Analysis")
        
        fig = px.pie(
            df, 
            values='count', 
            names=entity_col,
            title=f"Content Distribution by {entity_col.capitalize()}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Data Table")
        st.dataframe(df, use_container_width=True)
    
    # For time series data
    elif any(col in df.columns for col in ['date', 'time', 'year']):
        time_col = next(col for col in ['date', 'time', 'year'] if col in df.columns)
        numeric_cols = [col for col in df.columns if col not in [time_col] and pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            st.subheader("Temporal Analysis")
            fig = px.line(
                df, 
                x=time_col, 
                y=numeric_cols,
                title=f"Trend Analysis over {time_col.capitalize()}",
                labels={time_col: time_col.capitalize()}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("Data Table")
            st.dataframe(df, use_container_width=True)
    
    # For general data with numeric values
    else:
        # Find categorical and numeric columns
        categorical_cols = [col for col in df.columns if len(df[col].unique()) < 10]
        numeric_cols = [col for col in df.columns if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        if categorical_cols and numeric_cols:
            st.subheader("Data Analysis")
            selected_cat = st.selectbox("Select Category", categorical_cols)
            selected_num = st.selectbox("Select Value", numeric_cols)
            
            fig = px.bar(
                df,
                x=selected_cat,
                y=selected_num,
                color=selected_cat if len(categorical_cols) > 1 else None,
                title=f"{selected_num.capitalize()} by {selected_cat.capitalize()}",
                labels={selected_cat: selected_cat.capitalize(), selected_num: selected_num.capitalize()}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Data Table")
        st.dataframe(df, use_container_width=True)

# Sidebar for navigation and system status
with st.sidebar:
    st.markdown("<div class='main-header'>Hybrid RAG System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Neo4j + Vector Search</div>", unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### Navigation")
    page = st.radio(
        "Select a page",
        ["Question Answering", "Data Ingestion", "Database Explorer", "Advanced Analytics"],
        label_visibility="collapsed"
    )
    
    # System status
    st.markdown("### System Status")
    
    # Neo4j connection status
    neo4j_status = "✅ Connected" if st.session_state.neo4j_connected else "❌ Not Connected"
    st.markdown(f"**Neo4j:** {neo4j_status}")
    
    # Embedding model status
    embedding_status = "✅ Loaded" if st.session_state.get('embedding_model_loaded', False) else "❌ Not Loaded"
    st.markdown(f"**Embedding Model:** {embedding_status}")
    
    # Groq API status
    groq_status = "✅ Configured" if GROQ_API_KEY else "❌ Not Configured"
    st.markdown(f"**Groq API:** {groq_status}")
    
    if st.button("Check Connections"):
        with st.spinner("Checking Neo4j connection..."):
            st.session_state.neo4j_connected = test_connection()
        
        with st.spinner("Loading embedding model..."):
            try:
                st.session_state.embedding_model = load_embedding_model()
                st.session_state.embedding_model_loaded = True
            except Exception as e:
                st.session_state.embedding_model_loaded = False
                st.session_state.embedding_model_error = str(e)
        
        st.experimental_rerun()

# Main content
if page == "Question Answering":
    st.markdown("<div class='main-header'>Question Answering</div>", unsafe_allow_html=True)
    st.markdown("Ask questions about your data using natural language. The system will classify your query and use the appropriate retrieval method.")
    
    # Input for user query
    user_query = st.text_area("Enter your question:", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        show_debug = st.checkbox("Show Debug Info", value=False)
    with col2:
        if st.button("Submit Question", type="primary"):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                # Process the query
                with st.spinner("Processing your question..."):
                    try:
                        if show_debug:
                            # Get full response with debug info
                            response_dict = generate_final_answer(user_query)
                            
                            # Display the answer
                            st.markdown("### Answer")
                            st.markdown(response_dict['answer'])
                            
                            # Add to history
                            st.session_state.query_history.append({
                                "query": user_query,
                                "answer": response_dict['answer'],
                                "timestamp": time.time(),
                                "is_complex": response_dict['is_complex_query'],
                                "retrieval_method": response_dict['retrieval_method'],
                                "context": response_dict['context_used']
                            })
                            
                            # Display debug info
                            st.markdown("### Debug Information")
                            
                            st.markdown(f"**Query classified as:** {'Complex' if response_dict['is_complex_query'] else 'Simple'}")
                            st.markdown(f"**Retrieval method:** {response_dict['retrieval_method'].capitalize()}")
                            
                            # Show retrieved context
                            st.markdown("**Context used:**")
                            st.code(response_dict['context_used'], language="text")
                            
                        else:
                            # Get just the answer
                            answer = process_user_query(user_query, include_debug_info=False)
                            
                            # Display the answer
                            st.markdown("### Answer")
                            st.markdown(answer)
                            
                            # Add to history
                            st.session_state.query_history.append({
                                "query": user_query,
                                "answer": answer,
                                "timestamp": time.time()
                            })
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("### Recent Questions")
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Q: {item['query']}"):
                st.markdown(f"**Answer:**\n{item['answer']}")
                
                if 'is_complex' in item:
                    st.markdown(f"**Query type:** {'Complex' if item['is_complex'] else 'Simple'}")
                    st.markdown(f"**Retrieval method:** {item['retrieval_method'].capitalize()}")

elif page == "Data Ingestion":
    st.markdown("<div class='main-header'>Data Ingestion</div>", unsafe_allow_html=True)
    st.markdown("Upload and ingest data into your Neo4j database. The system supports text files, metadata files, and CSV files for model evaluations.")
    
    # Input for data paths
    st.markdown("### Data Source Paths")
    text_dir = st.text_input("Text Directory Path:", placeholder="/path/to/text/files")
    metadata_dir = st.text_input("Metadata Directory Path:", placeholder="/path/to/metadata/files")
    csv_dir = st.text_input("CSV Directory Path:", placeholder="/path/to/csv/files")
    
    if st.button("Start Ingestion", type="primary"):
        if not (text_dir or metadata_dir or csv_dir):
            st.warning("Please provide at least one directory path.")
        else:
            with st.spinner("Ingesting data..."):
                try:
                    ingest_all_data(
                        text_dir=text_dir if text_dir else None,
                        metadata_dir=metadata_dir if metadata_dir else None,
                        csv_dir=csv_dir if csv_dir else None
                    )
                    st.success("Data ingestion completed successfully!")
                except Exception as e:
                    st.error(f"Error during data ingestion: {str(e)}")

elif page == "Database Explorer":
    st.markdown("<div class='main-header'>Database Explorer</div>", unsafe_allow_html=True)
    st.markdown("Explore the content and structure of your Neo4j graph database.")
    
    # Get database statistics
    if st.button("Refresh Database Statistics"):
        with st.spinner("Fetching database statistics..."):
            st.session_state.db_stats = get_database_statistics()
    
    if 'db_stats' not in st.session_state:
        with st.spinner("Fetching database statistics..."):
            st.session_state.db_stats = get_database_statistics()
    
    stats = st.session_state.db_stats
    
    # Display database statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Node Counts")
        if stats['nodes']:
            fig = px.bar(
                x=list(stats['nodes'].keys()),
                y=list(stats['nodes'].values()),
                labels={'x': 'Node Type', 'y': 'Count'},
                color=list(stats['nodes'].keys())
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No nodes found in the database.")
    
    with col2:
        st.markdown("### Relationship Counts")
        if stats['relationships']:
            fig = px.bar(
                x=list(stats['relationships'].keys()),
                y=list(stats['relationships'].values()),
                labels={'x': 'Relationship Type', 'y': 'Count'},
                color=list(stats['relationships'].keys())
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No relationships found in the database.")
    
    # Custom Cypher query input
    st.markdown("### Custom Cypher Query")
    cypher_query = st.text_area(
        "Enter a custom Cypher query:",
        placeholder="MATCH (n) RETURN labels(n) as Label, count(n) as Count",
        height=100
    )
    
    if st.button("Run Cypher Query"):
        if not cypher_query:
            st.warning("Please enter a Cypher query.")
        else:
            with st.spinner("Executing query..."):
                try:
                    results = execute_cypher_query(cypher_query)
                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                    else:
                        st.info("Query executed successfully but returned no results.")
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")

elif page == "Advanced Analytics":
    st.markdown("<div class='main-header'>Advanced Analytics</div>", unsafe_allow_html=True)
    st.markdown("Run complex analytical queries and generate interactive dashboards.")
    
    # Query options
    query_type = st.selectbox(
        "Select Query Type:",
        ["Model Comparison", "Author Analysis", "Topic Distribution", "Custom Complex Query"]
    )
    
    if query_type == "Model Comparison":
        st.markdown("### Model Performance Comparison")
        metrics = ["MMLU", "GSM8K", "HumanEval", "All Metrics"]
        selected_metric = st.selectbox("Select Metric:", metrics)
        
        if st.button("Generate Dashboard", type="primary"):
            with st.spinner("Generating dashboard..."):
                try:
                    # Generate appropriate Cypher query
                    if selected_metric == "All Metrics":
                        cypher_query = """
                        MATCH (m:Model)-[:HAS_SCORE]->(s:Score)-[:FOR_METRIC]->(e:Metric)
                        RETURN m.name AS model, e.name AS metric, s.value AS score
                        ORDER BY m.name, e.name
                        """
                    else:
                        cypher_query = f"""
                        MATCH (m:Model)-[:HAS_SCORE]->(s:Score)-[:FOR_METRIC]->(e:Metric {{name: '{selected_metric}'}})
                        RETURN m.name AS model, e.name AS metric, s.value AS score
                        ORDER BY score DESC
                        """
                    
                    # Execute query
                    results = execute_cypher_query(cypher_query)
                    
                    if results:
                        df = pd.DataFrame(results)
                        render_dashboard(df, "comparison")
                    else:
                        st.info("No model performance data found.")
                except Exception as e:
                    st.error(f"Error generating dashboard: {str(e)}")
    
    elif query_type == "Author Analysis":
        st.markdown("### Author Contribution Analysis")
        topic_filter = st.text_input("Filter by Topic (optional):", placeholder="AI, NLP, etc.")
        
        if st.button("Generate Dashboard", type="primary"):
            with st.spinner("Generating dashboard..."):
                try:
                    # Generate appropriate Cypher query
                    if topic_filter:
                        topics = [t.strip() for t in topic_filter.split(',')]
                        cypher_query = """
                        MATCH (a:Author)<-[:HAS_AUTHOR]-(c:Chunk)-[:HAS_TOPIC]->(t:Topic)
                        WHERE t.name IN $topics
                        RETURN a.name AS author, count(DISTINCT c) AS count
                        ORDER BY count DESC
                        """
                        results = execute_cypher_query(cypher_query, {"topics": topics})
                    else:
                        cypher_query = """
                        MATCH (a:Author)<-[:HAS_AUTHOR]-(c:Chunk)
                        RETURN a.name AS author, count(c) AS count
                        ORDER BY count DESC
                        """
                        results = execute_cypher_query(cypher_query)
                    
                    if results:
                        df = pd.DataFrame(results)
                        render_dashboard(df, "author")
                    else:
                        st.info("No author data found.")
                except Exception as e:
                    st.error(f"Error generating dashboard: {str(e)}")
    
    elif query_type == "Topic Distribution":
        st.markdown("### Topic Distribution Analysis")
        author_filter = st.text_input("Filter by Author (optional):", placeholder="Author name")
        
        if st.button("Generate Dashboard", type="primary"):
            with st.spinner("Generating dashboard..."):
                try:
                    # Generate appropriate Cypher query
                    if author_filter:
                        cypher_query = """
                        MATCH (t:Topic)<-[:HAS_TOPIC]-(c:Chunk)-[:HAS_AUTHOR]->(a:Author {name: $author})
                        RETURN t.name AS topic, count(c) AS count
                        ORDER BY count DESC
                        """
                        results = execute_cypher_query(cypher_query, {"author": author_filter})
                    else:
                        cypher_query = """
                        MATCH (t:Topic)<-[:HAS_TOPIC]-(c:Chunk)
                        RETURN t.name AS topic, count(c) AS count
                        ORDER BY count DESC
                        """
                        results = execute_cypher_query(cypher_query)
                    
                    if results:
                        df = pd.DataFrame(results)
                        render_dashboard(df, "topic")
                    else:
                        st.info("No topic data found.")
                except Exception as e:
                    st.error(f"Error generating dashboard: {str(e)}")
    
    elif query_type == "Custom Complex Query":
        st.markdown("### Custom Complex Query")

        # Natural language query input
        nl_query = st.text_area(
            "Enter your analytical question in natural language:",
            placeholder="Compare the performance of different models on the MMLU benchmark",
            height=100
        )

        # Option to see the generated Cypher query
        show_cypher = st.checkbox("Show generated Cypher query", value=True)

        if st.button("Generate Dashboard", type="primary"):
            if not nl_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Processing your query..."):
                    try:
                        # Generate Cypher query from natural language
                        cypher_query = generate_cypher_query(nl_query)
                        
                        if show_cypher:
                            st.markdown("### Generated Cypher Query")
                            st.code(cypher_query, language="cypher")
                        
                        # Execute the query
                        results = execute_cypher_query(cypher_query)
                        
                        if results:
                            df = process_query_results(results, "custom")
                            if df is not None and not df.empty:
                                render_dashboard(df, "custom")
                            else:
                                st.markdown("### Query Results")
                                st.dataframe(pd.DataFrame(results), use_container_width=True)
                        else:
                            st.info("Query executed successfully but returned no results.")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")