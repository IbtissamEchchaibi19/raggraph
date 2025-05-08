import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
NEO4J_URI = "neo4j+s://cec92046.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "R4lh9gXE91brQEQASYKND94rgBeX1_Het8FINTKK12E"
DATA_DIR = "C:/Users/ibtis/OneDrive/Desktop/graphrag/Data"
GEMINI_API_KEY = "AIzaSyAwG0wFj5kFHU89Y7Mvimql7MB2sxVxc-s"  # Replace with your actual API key

# Initialize Hugging Face model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Neo4j connection
def get_neo4j_connection():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to load data from files
def load_data():
    all_data = []
    
    # Walk through the data directory
    for root, dirs, files in os.walk(DATA_DIR):
        report_data = {}
        
        # Process each file in the current subdirectory
        for file in files:
            file_path = os.path.join(root, file)
            
            # Process JSON metadata files
            if file.endswith('.json'):
                with open(file_path, 'r') as f:
                    report_data['metadata'] = json.load(f)
                    # Use report number as identifier
                    report_data['report_id'] = report_data['metadata'].get('report_number', os.path.basename(root))
            
            # Process TXT content files
            elif file.endswith('.txt'):
                with open(file_path, 'r') as f:
                    report_data['text_content'] = f.read()
            
            # Process CSV parameter files
            elif file.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    report_data['parameters'] = df.to_dict('records')
                except Exception as e:
                    print(f"Error reading CSV file {file_path}: {e}")
        
        # If we found all necessary components, add to our collection
        if all(key in report_data for key in ['metadata', 'text_content', 'parameters']):
            all_data.append(report_data)
    
    return all_data

# Function to create embeddings
def create_embeddings(text):
    return embedding_model.encode(text).tolist()

# Function to build the Neo4j graph
def build_knowledge_graph(data):
    driver = get_neo4j_connection()
    
    with driver.session() as session:
        # Clear existing graph
        session.run("MATCH (n) DETACH DELETE n")
        
        for report in data:
            # Create Report node
            report_id = report['report_id']
            metadata = report['metadata']
            text_content = report['text_content']
            
            # Extract and generate embeddings for chunks of text content
            text_chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]
            
            # Create Report node with metadata
            metadata_properties = ', '.join([f"{k}: $metadata.{k}" for k in metadata.keys()])
            query = f"""
            CREATE (r:Report {{report_id: $report_id, text_embedding: $text_embedding, {metadata_properties}}})
            RETURN r
            """
            
            # Create an embedding of the full text for the Report node
            text_embedding = create_embeddings(text_content)
            
            session.run(query, {
                'report_id': report_id,
                'text_embedding': text_embedding,
                'metadata': metadata
            })
            
            # Create TextChunk nodes linked to the Report
            for i, chunk in enumerate(text_chunks):
                chunk_embedding = create_embeddings(chunk)
                session.run("""
                MATCH (r:Report {report_id: $report_id})
                CREATE (t:TextChunk {chunk_id: $chunk_id, content: $content, embedding: $embedding})
                CREATE (r)-[:HAS_CHUNK]->(t)
                """, {
                    'report_id': report_id,
                    'chunk_id': f"{report_id}_chunk_{i}",
                    'content': chunk,
                    'embedding': chunk_embedding
                })
            
            # Create Parameter nodes and relationships
            for param in report['parameters']:
                # Build a dictionary for parameter mapping that Neo4j can use safely
                param_dict = {}
                cypher_parts = []
                
                # Create safe property assignments
                for k, v in param.items():
                    # Create a safe parameter reference
                    safe_key = f"param_{len(param_dict)}"
                    param_dict[safe_key] = v
                    # Add to Cypher parts with backtick-escaped property name
                    cypher_parts.append(f"`{k}`: ${safe_key}")
                
                # Join all property assignments
                param_props = ", ".join(cypher_parts)
                
                # Add the report_id to our parameters
                param_dict['report_id'] = report_id
                
                # Execute the Cypher query with our safely constructed property list
                cypher_query = f"""
                MATCH (r:Report {{report_id: $report_id}})
                CREATE (p:Parameter {{{param_props}}})
                CREATE (r)-[:HAS_PARAMETER]->(p)
                """
                
                session.run(cypher_query, param_dict)
    
    # Create indexes for better performance
    with driver.session() as session:
        try:
            # Try the Neo4j 4.x+ syntax first
            session.run("CREATE INDEX FOR (r:Report) ON (r.report_id)")
            # Try to create index on Parameter nodes if there's a name property
            session.run("CREATE INDEX FOR (p:Parameter) ON (p.name)")
        except Exception as e:
            # Fall back to older Neo4j syntax
            try:
                print(f"Trying older Neo4j syntax for indexes due to: {e}")
                session.run("CREATE INDEX ON :Report(report_id)")
                session.run("CREATE INDEX ON :Parameter(name)")
            except Exception as e2:
                print(f"Warning: Could not create indexes: {e2}")
    
    driver.close()

# Function to query the graph database
def query_graph(query_text):
    # Extract key information from the query
    report_number_match = re.search(r'report (?:number|#)?\s*(\w+)', query_text, re.IGNORECASE)
    date_match = re.search(r'(?:on|date)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})', query_text, re.IGNORECASE)
    region_match = re.search(r'(?:in|region|area)\s*(\w+)', query_text, re.IGNORECASE)
    param_match = re.search(r'(?:the|value of|about|for)\s*(pH|temperature|conductivity)', query_text, re.IGNORECASE)
    
    # Connect to Neo4j
    driver = get_neo4j_connection()
    results = []
    
    try:
        with driver.session() as session:
            # Case 1: Query for specific report number
            if report_number_match:
                report_number = report_number_match.group(1)
                result = session.run("""
                MATCH (r:Report {report_id: $report_id})
                OPTIONAL MATCH (r)-[:HAS_PARAMETER]->(p)
                RETURN r, collect(p) as parameters
                """, {'report_id': report_number})
                
                for record in result:
                    report = record["r"]
                    params = record["parameters"]
                    results.append({
                        'report': dict(report),
                        'parameters': [dict(p) for p in params]
                    })
            
            # Case 2: Query for specific date
            elif date_match:
                date_str = date_match.group(1)
                # Convert various date formats to a standard one
                try:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, '%m-%d-%Y')
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(date_str, '%B %d, %Y')
                        except ValueError:
                            date_obj = None
                
                if date_obj:
                    standard_date = date_obj.strftime('%Y-%m-%d')
                    result = session.run("""
                    MATCH (r:Report)
                    WHERE r.date = $date
                    OPTIONAL MATCH (r)-[:HAS_PARAMETER]->(p)
                    RETURN r, collect(p) as parameters
                    """, {'date': standard_date})
                    
                    for record in result:
                        report = record["r"]
                        params = record["parameters"]
                        results.append({
                            'report': dict(report),
                            'parameters': [dict(p) for p in params]
                        })
            
            # Case 3: Query for specific parameter
            elif param_match:
                param_name = param_match.group(1).lower()
                result = session.run("""
                MATCH (r:Report)-[:HAS_PARAMETER]->(p)
                WHERE toLower(p.name) = $param_name
                RETURN r, p
                """, {'param_name': param_name})
                
                for record in result:
                    results.append({
                        'report': dict(record["r"]),
                        'parameter': dict(record["p"])
                    })
            
            # Case 4: Query by region
            elif region_match:
                region = region_match.group(1)
                result = session.run("""
                MATCH (r:Report)
                WHERE toLower(r.region) = toLower($region)
                OPTIONAL MATCH (r)-[:HAS_PARAMETER]->(p)
                RETURN r, collect(p) as parameters
                """, {'region': region})
                
                for record in result:
                    report = record["r"]
                    params = record["parameters"]
                    results.append({
                        'report': dict(report),
                        'parameters': [dict(p) for p in params]
                    })
            
            # Case 5: Semantic search using embeddings
            else:
                query_embedding = create_embeddings(query_text)
                
                # First, try to use APOC dot product if available
                try:
                    result = session.run("""
                    MATCH (r:Report)
                    WHERE r.text_embedding IS NOT NULL
                    WITH r, apoc.coll.dotProduct(r.text_embedding, $query_embedding) AS similarity
                    WHERE similarity > 0.5
                    RETURN r, similarity
                    ORDER BY similarity DESC
                    LIMIT 5
                    """, {'query_embedding': query_embedding})
                    
                    records = list(result)
                    if records:  # If APOC worked, use these results
                        for record in records:
                            results.append({
                                'report': dict(record["r"]),
                                'similarity': record["similarity"]
                            })
                    else:
                        # If no results or APOC not available, fall back to Python-based similarity
                        raise Exception("No results from APOC function")
                        
                except Exception as e:
                    print(f"APOC method failed: {e}. Falling back to Python-based similarity")
                    # Fall back to Python-based similarity calculation
                    result = session.run("""
                    MATCH (r:Report)
                    WHERE r.text_embedding IS NOT NULL
                    RETURN r
                    """)
                    
                    # Collect all reports and calculate similarity in Python
                    reports = []
                    embeddings = []
                    for record in result:
                        report = dict(record["r"])
                        if "text_embedding" in report:
                            reports.append(report)
                            embeddings.append(report["text_embedding"])
                    
                    if embeddings:
                        # Convert query embedding to numpy array and reshape
                        query_embedding_np = np.array(query_embedding).reshape(1, -1)
                        # Convert report embeddings to numpy array
                        embeddings_np = np.array(embeddings)
                        
                        # Calculate cosine similarities
                        similarities = cosine_similarity(query_embedding_np, embeddings_np)[0]
                        
                        # Create a list of (report, similarity) pairs
                        report_similarities = list(zip(reports, similarities))
                        
                        # Sort by similarity (descending) and take top 5
                        report_similarities.sort(key=lambda x: x[1], reverse=True)
                        top_reports = report_similarities[:5]
                        
                        # Add to results
                        for report, similarity in top_reports:
                            if similarity > 0.5:  # Only include results above threshold
                                results.append({
                                    'report': report,
                                    'similarity': float(similarity)
                                })
    
    finally:
        driver.close()
    
    return results

# Function to generate response using Gemini
def generate_response(query, results):
    if not results:
        return "I couldn't find any information related to your query in the laboratory reports."
    
    # Format context from the retrieved results
    context = "Here is the information I found in the laboratory reports:\n\n"
    
    for i, result in enumerate(results):
        context += f"Report {i+1}:\n"
        if 'report' in result:
            context += f"Report ID: {result['report'].get('report_id', 'Unknown')}\n"
            
            # Add relevant metadata
            for key, value in result['report'].items():
                if key not in ['report_id', 'text_embedding'] and isinstance(value, (str, int, float)):
                    context += f"{key.capitalize()}: {value}\n"
        
        # Add parameters if available
        if 'parameters' in result and result['parameters']:
            context += "Parameters:\n"
            for param in result['parameters']:
                for key, value in param.items():
                    context += f"- {key}: {value}\n"
        
        # Add individual parameter if available
        if 'parameter' in result:
            context += "Parameter details:\n"
            for key, value in result['parameter'].items():
                context += f"- {key}: {value}\n"
        
        context += "\n"
    
    # Create the prompt for Gemini
    prompt = f"""
    Based on the following information from laboratory reports:
    
    {context}
    
    Please provide a concise, accurate answer to this question: "{query}"
    Stick to the facts in the provided information. If you don't have enough information, say so.
    """
    
    # Generate response using Gemini
    response = gemini_model.generate_content(prompt)
    return response.text

# Function to create a dashboard with Streamlit
def create_dashboard():
    st.title("Laboratory Reports Analysis Dashboard")
    
    # Load all data
    all_data = load_data()
    
    # Sidebar for filtering
    st.sidebar.header("Filters")
    
    # Extract all available report IDs
    report_ids = [report['report_id'] for report in all_data]
    selected_reports = st.sidebar.multiselect("Select Reports", report_ids, default=report_ids[:min(3, len(report_ids))])
    
    # Filter data based on selection
    filtered_data = [report for report in all_data if report['report_id'] in selected_reports]
    
    # Display debug information about parameters
    if filtered_data:
        st.sidebar.subheader("Data Overview")
        for i, report in enumerate(filtered_data):
            if 'parameters' in report and report['parameters']:
                st.sidebar.write(f"Report {report['report_id']} parameter sample:")
                param_sample = report['parameters'][0]
                param_info = {k: f"{v} ({type(v).__name__})" for k, v in param_sample.items()}
                st.sidebar.json(param_info)
                break
    
    # Extract parameter types from the first report (assuming consistent schema)
    if filtered_data and 'parameters' in filtered_data[0] and filtered_data[0]['parameters']:
        parameter_keys = list(filtered_data[0]['parameters'][0].keys())
        
        # Only offer numeric parameters for visualization
        numeric_params = []
        for report in filtered_data:
            if 'parameters' in report and report['parameters']:
                for param_key in parameter_keys:
                    for param in report['parameters']:
                        if param_key in param and isinstance(param[param_key], (int, float)):
                            if param_key not in numeric_params:
                                numeric_params.append(param_key)
                            break
        
        if numeric_params:
            selected_param = st.sidebar.selectbox("Select Parameter to Visualize", numeric_params)
        else:
            selected_param = None
            st.sidebar.warning("No numeric parameters found for visualization.")
    else:
        selected_param = None
        st.sidebar.warning("No parameters found in the selected reports.")
    
    # Display individual report visualizations
    st.header("Individual Report Analysis")
    
    for report in filtered_data:
        st.subheader(f"Report: {report['report_id']}")
        
        # Display metadata
        with st.expander("Metadata"):
            st.json(report['metadata'])
        
        # Display parameters as table
        with st.expander("Parameters Table"):
            if 'parameters' in report and report['parameters']:
                st.dataframe(pd.DataFrame(report['parameters']))
        
        # Create visualization for selected parameter
        if selected_param and 'parameters' in report and report['parameters']:
            try:
                # Convert parameters to DataFrame
                param_data = pd.DataFrame(report['parameters'])
                
                # Check if selected parameter exists and has numeric data
                if selected_param in param_data.columns:
                    # Try to convert to numeric, coercing errors to NaN
                    param_data[selected_param] = pd.to_numeric(param_data[selected_param], errors='coerce')
                    
                    # Check if we have valid numeric data after conversion
                    if not param_data[selected_param].isna().all():
                        # Create a matplotlib figure
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot based on available columns
                        if 'date' in param_data.columns or 'time' in param_data.columns:
                            x_col = 'date' if 'date' in param_data.columns else 'time'
                            # For date/time columns, convert to datetime if possible
                            try:
                                param_data[x_col] = pd.to_datetime(param_data[x_col], errors='coerce')
                                param_data = param_data.sort_values(by=x_col)
                            except:
                                pass
                            
                            # Line plot for time series data
                            param_data.plot(x=x_col, y=selected_param, kind='line', ax=ax, marker='o')
                        else:
                            # Create index-based bar chart
                            valid_data = param_data[~param_data[selected_param].isna()]
                            valid_data[selected_param].plot(kind='bar', ax=ax)
                        
                        ax.set_title(f"{selected_param} in Report {report['report_id']}")
                        ax.set_ylabel(selected_param)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info(f"No valid numeric data for '{selected_param}' in this report.")
                else:
                    st.info(f"'{selected_param}' not found in this report's parameters.")
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Please check the Parameters Table to see the actual data.")
    
    # Cross-report comparison
    if len(filtered_data) > 1 and selected_param:
        st.header("Cross-Report Comparison")
        
        try:
            # Collect data from all selected reports
            comparison_data = []
            for report in filtered_data:
                if 'parameters' in report and report['parameters']:
                    for param in report['parameters']:
                        if selected_param in param:
                            try:
                                # Try to convert to numeric value
                                param_value = pd.to_numeric(param[selected_param], errors='coerce')
                                if not pd.isna(param_value):  # Only add if conversion was successful
                                    comparison_data.append({
                                        'report_id': report['report_id'],
                                        selected_param: param_value,
                                        'region': report['metadata'].get('region', 'Unknown')
                                    })
                            except:
                                # Skip non-numeric values
                                pass
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Check if we have enough data for visualization
                if len(comparison_df) > 0 and not comparison_df[selected_param].isna().all():
                    # Create a bar chart comparing the selected parameter across reports
                    fig, ax = plt.subplots(figsize=(12, 6))
                    comparison_df.plot(x='report_id', y=selected_param, kind='bar', ax=ax)
                    ax.set_title(f"Comparison of {selected_param} Across Reports")
                    ax.set_ylabel(selected_param)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Check if we have region data for box plot
                    if 'region' in comparison_df.columns and comparison_df['region'].nunique() > 1:
                        # Create a box plot to show distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        comparison_df.boxplot(column=selected_param, by='region', ax=ax)
                        plt.suptitle("")  # Remove default title
                        ax.set_title(f"Distribution of {selected_param} by Region")
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Show the data table
                    st.subheader("Comparison Data")
                    st.dataframe(comparison_df)
                else:
                    st.info(f"No valid numeric data for '{selected_param}' across reports.")
            else:
                st.info(f"No valid data for '{selected_param}' found across reports.")
        
        except Exception as e:
            st.error(f"Error in cross-report comparison: {str(e)}")
            st.info("Try selecting different reports or parameters.")

# Main chatbot function
def chatbot():
    st.title("Laboratory Reports Chatbot")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask about laboratory reports"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query the graph
                results = query_graph(prompt)
                
                # Generate response using Gemini
                response = generate_response(prompt, results)
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main application
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Data Loading", "Chatbot", "Dashboard"])
    
    if page == "Data Loading":
        st.title("Data Loading and Graph Building")
        if st.button("Load Data and Build Knowledge Graph"):
            with st.spinner("Loading data..."):
                data = load_data()
                st.success(f"Loaded {len(data)} laboratory reports")
            
            with st.spinner("Building knowledge graph..."):
                build_knowledge_graph(data)
                st.success("Knowledge graph built successfully!")
    
    elif page == "Chatbot":
        chatbot()
    
    elif page == "Dashboard":
        create_dashboard()

if __name__ == "__main__":
    main()