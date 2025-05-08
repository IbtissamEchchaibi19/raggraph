import os
import json
import re
import numpy as np
import streamlit as st
from datetime import datetime
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
NEO4J_URI = "neo4j+s://e8beaade.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "oQD-Scg83b-vuRc0fWEqWUQgQCzF-BXGOyp-DraBxek"
DATA_DIR = "C:/Users/ibtis/OneDrive/Desktop/graphrag/Data"
GEMINI_API_KEY = "AIzaSyAwG0wFj5kFHU89Y7Mvimql7MB2sxVxc-s"

# Initialize Hugging Face model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

class GraphRAGSystem:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.embedding_model = load_embedding_model()
    
    def close(self):
        self.driver.close()
    
    def check_graph_populated(self):
        """Check if the Neo4j database already has data"""
        with self.driver.session() as session:
            result = session.run("MATCH (r:Report) RETURN count(r) as count")
            record = result.single()
            return record and record["count"] > 0

    def load_data(self):
        """Load data from files in the DATA_DIR directory"""
        all_data = []
        
        # Walk through the data directory
        for root, dirs, files in os.walk(DATA_DIR):
            report_data = {}
            
            # Process each file in the current subdirectory
            for file in files:
                file_path = os.path.join(root, file)
                
                # Process JSON metadata files
                if file.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        report_data['metadata'] = json.load(f)
                        # Use report number as identifier
                        report_data['report_id'] = report_data['metadata'].get('report_number', os.path.basename(root))
                
                # Process TXT content files
                elif file.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        report_data['text_content'] = f.read()
            
            # If we found necessary components, add to our collection
            if all(key in report_data for key in ['metadata', 'text_content']):
                all_data.append(report_data)
                st.write(f"Loaded report {report_data['report_id']}")
            
        st.write(f"Loaded {len(all_data)} reports in total")
        return all_data
    
    def create_embeddings(self, text):
        """Create embeddings for text using the SentenceTransformer model"""
        return self.embedding_model.encode(text).tolist()
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into chunks with overlap for better context preservation"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= chunk_size // 2:  # Only keep chunks that are reasonably sized
                chunks.append(chunk)
        return chunks
    
    def build_knowledge_graph(self, data):
        """Build the Neo4j knowledge graph from the loaded data"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.write("Building knowledge graph...")
        
        with self.driver.session() as session:
            # Clear existing graph
            session.run("MATCH (n) DETACH DELETE n")
            
            total_reports = len(data)
            for i, report in enumerate(data):
                # Update progress
                progress = (i + 1) / total_reports
                progress_bar.progress(progress)
                status_text.write(f"Processing report {i+1} of {total_reports}: {report['report_id']}")
                
                # Extract data
                report_id = report['report_id']
                metadata = report['metadata']
                text_content = report['text_content']
                
                # Create embeddings for the full text
                text_embedding = self.create_embeddings(text_content)
                
                # Create Report node with metadata
                # Build a dictionary with all metadata properties
                report_properties = {
                    'report_id': report_id,
                    'text_embedding': text_embedding
                }
                
                # Add all metadata as properties
                for key, value in metadata.items():
                    # Handle arrays by converting them to strings
                    if isinstance(value, list):
                        report_properties[key] = json.dumps(value)
                    else:
                        report_properties[key] = value
                
                # Create the Report node
                session.run(
                    "CREATE (r:Report $properties)",
                    properties=report_properties
                )
                
                # Process and create chunks
                text_chunks = self.chunk_text(text_content)
                
                # Create TextChunk nodes linked to the Report
                for i, chunk in enumerate(text_chunks):
                    chunk_embedding = self.create_embeddings(chunk)
                    session.run("""
                    MATCH (r:Report {report_id: $report_id})
                    CREATE (t:TextChunk {
                        chunk_id: $chunk_id,
                        content: $content,
                        embedding: $embedding
                    })
                    CREATE (r)-[:HAS_CHUNK]->(t)
                    """, {
                        'report_id': report_id,
                        'chunk_id': f"{report_id}_chunk_{i}",
                        'content': chunk,
                        'embedding': chunk_embedding
                    })
            
            # Create indexes for better performance
            try:
                # Try creating constraints for uniqueness
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Report) REQUIRE r.report_id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.chunk_id IS UNIQUE")
                # Create indexes for properties commonly used in queries
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:Report) ON (r.client_name)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:Report) ON (r.report_date)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (r:Report) ON (r.sample_id)")
            except Exception as e:
                st.warning(f"Could not create all indexes or constraints: {e}")
                # Try older Neo4j syntax for compatibility
                try:
                    session.run("CREATE INDEX ON :Report(report_id)")
                    session.run("CREATE INDEX ON :Report(client_name)")
                    session.run("CREATE INDEX ON :Report(report_date)")
                except Exception as e2:
                    st.warning(f"Could not create indexes with old syntax: {e2}")
        
        status_text.write("Knowledge graph built successfully!")
        progress_bar.empty()
    
    def query_graph(self, query_text):
        """Query the Neo4j graph based on the query text"""
        st.info(f"Processing query: {query_text}")
        
        # Analyze query for key information
        report_number_match = re.search(r'report (?:number|#)?\s*(\w+)', query_text, re.IGNORECASE)
        sample_id_match = re.search(r'sample (?:id|number|#)?\s*(\w+)', query_text, re.IGNORECASE)
        client_match = re.search(r'(?:client|company|farm)\s*([\w\s]+)', query_text, re.IGNORECASE)
        date_match = re.search(r'(?:on|date|between)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})', query_text, re.IGNORECASE)
        date_range_match = re.search(r'between\s*(\w+ \d{1,2},? \d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*and\s*(\w+ \d{1,2},? \d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', query_text, re.IGNORECASE)
        certification_match = re.search(r'certified\s*(?:under|with)?\s*([\w\s\/]+)', query_text, re.IGNORECASE)
        parameter_match = re.search(r'(?:parameter|value|show|temperature|pH|conductivity)\s*([\w\s]+)', query_text, re.IGNORECASE)
        
        results = []
        
        try:
            with self.driver.session() as session:
                # Case 1: Query for specific report number
                if report_number_match:
                    report_number = report_number_match.group(1)
                    result = session.run("""
                    MATCH (r:Report {report_id: $report_id})
                    RETURN r
                    """, {'report_id': report_number})
                    
                    for record in result:
                        report = dict(record["r"])
                        results.append({'report': report})
                
                # Case 2: Query for specific sample ID
                elif sample_id_match:
                    sample_id = sample_id_match.group(1)
                    result = session.run("""
                    MATCH (r:Report {sample_id: $sample_id})
                    RETURN r
                    """, {'sample_id': sample_id})
                    
                    for record in result:
                        report = dict(record["r"])
                        results.append({'report': report})
                
                # Case 3: Query for client name
                elif client_match:
                    client_name = client_match.group(1).strip()
                    result = session.run("""
                    MATCH (r:Report)
                    WHERE r.client_name CONTAINS $client_name
                    RETURN r
                    """, {'client_name': client_name})
                    
                    for record in result:
                        report = dict(record["r"])
                        results.append({'report': report})
                
                # Case 4: Query for specific date
                elif date_match and not date_range_match:
                    date_str = date_match.group(1)
                    # Try different date formats
                    try:
                        # Convert to standard format
                        formatted_date = self.normalize_date(date_str)
                        
                        result = session.run("""
                        MATCH (r:Report)
                        WHERE r.report_date CONTAINS $date 
                           OR r.sample_received_date CONTAINS $date
                           OR r.analysis_start_date CONTAINS $date
                           OR r.analysis_end_date CONTAINS $date
                        RETURN r
                        """, {'date': formatted_date})
                        
                        for record in result:
                            report = dict(record["r"])
                            results.append({'report': report})
                    except:
                        pass
                
                # Case 5: Query for date range
                elif date_range_match:
                    start_date_str = date_range_match.group(1)
                    end_date_str = date_range_match.group(2)
                    
                    try:
                        # Convert to datetime objects for comparison
                        start_date = self.parse_date(start_date_str)
                        end_date = self.parse_date(end_date_str)
                        
                        # Get all reports and filter in Python (simpler than complex Cypher date handling)
                        result = session.run("MATCH (r:Report) RETURN r")
                        
                        for record in result:
                            report = dict(record["r"])
                            report_date = None
                            
                            # Try to parse the report date
                            if 'report_date' in report:
                                try:
                                    report_date = self.parse_date(report['report_date'])
                                    if start_date <= report_date <= end_date:
                                        results.append({'report': report})
                                except:
                                    pass
                    except:
                        pass
                
                # Case 6: Query for certification
                elif certification_match:
                    certification = certification_match.group(1).strip()
                    
                    result = session.run("""
                    MATCH (r:Report)
                    WHERE r.certifications CONTAINS $certification
                    RETURN r
                    """, {'certification': certification})
                    
                    for record in result:
                        report = dict(record["r"])
                        results.append({'report': report})
                
                # Case 7: Semantic search using embeddings
                else:
                    # Create embedding for the query
                    query_embedding = self.create_embeddings(query_text)
                    
                    # Try to use vector similarity in Neo4j
                    try:
                        # First try to get relevant reports using full-text embedding
                        result = session.run("""
                        MATCH (r:Report)
                        WHERE r.text_embedding IS NOT NULL
                        WITH r, gds.similarity.cosine(r.text_embedding, $query_embedding) AS similarity
                        WHERE similarity > 0.6
                        RETURN r, similarity
                        ORDER BY similarity DESC
                        LIMIT 3
                        """, {'query_embedding': query_embedding})
                        
                        for record in result:
                            report = dict(record["r"])
                            similarity = record["similarity"]
                            results.append({
                                'report': report,
                                'similarity': similarity,
                                'type': 'report_match'
                            })
                        
                        # Then try to get specific chunks that might be more relevant
                        chunk_result = session.run("""
                        MATCH (r:Report)-[:HAS_CHUNK]->(t:TextChunk)
                        WHERE t.embedding IS NOT NULL
                        WITH t, r, gds.similarity.cosine(t.embedding, $query_embedding) AS similarity
                        WHERE similarity > 0.65
                        RETURN t, r, similarity
                        ORDER BY similarity DESC
                        LIMIT 5
                        """, {'query_embedding': query_embedding})
                        
                        for record in chunk_result:
                            chunk = dict(record["t"])
                            report = dict(record["r"])
                            similarity = record["similarity"]
                            results.append({
                                'chunk': chunk,
                                'report': report,
                                'similarity': similarity,
                                'type': 'chunk_match'
                            })
                    
                    except Exception as e:
                        st.warning(f"Neo4j vector similarity failed: {e}, falling back to Python")
                        
                        # Fall back to Python-based cosine similarity
                        # Get all reports
                        result = session.run("""
                        MATCH (r:Report)
                        WHERE r.text_embedding IS NOT NULL
                        RETURN r
                        """)
                        
                        reports = []
                        embeddings = []
                        for record in result:
                            report = dict(record["r"])
                            if "text_embedding" in report:
                                reports.append(report)
                                embeddings.append(report["text_embedding"])
                        
                        if embeddings:
                            # Calculate similarities
                            query_embedding_np = np.array(query_embedding).reshape(1, -1)
                            embeddings_np = np.array(embeddings)
                            similarities = cosine_similarity(query_embedding_np, embeddings_np)[0]
                            
                            # Get top matches
                            report_similarities = list(zip(reports, similarities))
                            report_similarities.sort(key=lambda x: x[1], reverse=True)
                            top_reports = report_similarities[:3]
                            
                            for report, similarity in top_reports:
                                if similarity > 0.6:
                                    results.append({
                                        'report': report,
                                        'similarity': float(similarity),
                                        'type': 'report_match'
                                    })
                        
                        # Get chunks
                        chunk_result = session.run("""
                        MATCH (r:Report)-[:HAS_CHUNK]->(t:TextChunk)
                        WHERE t.embedding IS NOT NULL
                        RETURN t, r
                        """)
                        
                        chunks = []
                        chunk_reports = []
                        chunk_embeddings = []
                        
                        for record in chunk_result:
                            chunk = dict(record["t"])
                            report = dict(record["r"])
                            if "embedding" in chunk:
                                chunks.append(chunk)
                                chunk_reports.append(report)
                                chunk_embeddings.append(chunk["embedding"])
                        
                        if chunk_embeddings:
                            # Calculate similarities
                            chunk_embeddings_np = np.array(chunk_embeddings)
                            chunk_similarities = cosine_similarity(query_embedding_np, chunk_embeddings_np)[0]
                            
                            # Get top matches
                            chunk_data = list(zip(chunks, chunk_reports, chunk_similarities))
                            chunk_data.sort(key=lambda x: x[2], reverse=True)
                            top_chunks = chunk_data[:5]
                            
                            for chunk, report, similarity in top_chunks:
                                if similarity > 0.65:
                                    results.append({
                                        'chunk': chunk,
                                        'report': report,
                                        'similarity': float(similarity),
                                        'type': 'chunk_match'
                                    })
        
        except Exception as e:
            st.error(f"Error querying graph: {e}")
        
        st.write(f"Found {len(results)} results")
        return results
    
    def normalize_date(self, date_str):
        """Convert various date formats to a standard format"""
        if '.' in date_str:  # Format like dd.mm.yyyy
            parts = date_str.split('.')
            if len(parts) == 3:
                day, month, year = parts
                if len(year) == 2:
                    year = f"20{year}"
                return f"{day}.{month}.{year}"
        
        # Try to parse with datetime
        return self.parse_date(date_str).strftime("%d.%m.%Y")
    
    def parse_date(self, date_str):
        """Parse different date formats to datetime object"""
        formats = [
            "%d.%m.%Y",  # 29.07.2022
            "%Y-%m-%d",  # 2022-07-29
            "%m/%d/%Y",  # 07/29/2022
            "%B %d, %Y", # July 29, 2022
            "%b %d, %Y", # Jul 29, 2022
            "%d-%m-%Y",  # 29-07-2022
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        raise ValueError(f"Could not parse date: {date_str}")
    
    def generate_response(self, query, results):
        """Generate response using Gemini based on query and results"""
        if not results:
            return "I couldn't find any information related to your query in the laboratory reports."
        
        # Format context from the retrieved results
        context = "Here is the information I found in the laboratory reports:\n\n"
        
        for i, result in enumerate(results):
            # Add report information
            if 'report' in result:
                report = result['report']
                context += f"Report {i+1}: {report.get('report_id', 'Unknown ID')}\n"
                
                # Add metadata fields that might be relevant
                important_fields = [
                    'client_name', 'report_date', 'sample_id', 'sample_received_date',
                    'analysis_start_date', 'analysis_end_date', 'certifications'
                ]
                
                for field in important_fields:
                    if field in report and report[field]:
                        # Handle certifications as a special case
                        if field == 'certifications' and isinstance(report[field], str):
                            try:
                                certs = json.loads(report[field])
                                context += f"Certifications: {', '.join(certs)}\n"
                            except:
                                context += f"Certifications: {report[field]}\n"
                        else:
                            context += f"{field.replace('_', ' ').title()}: {report[field]}\n"
            
            # Add chunk information for semantic searches
            if 'chunk' in result:
                chunk = result['chunk']
                context += f"Relevant content: {chunk.get('content', '')}\n"
            
            context += "\n"
        
        # Create the prompt for Gemini
        prompt = f"""
        Based on the following information from laboratory reports:
        
        {context}
        
        Please provide a concise, accurate answer to this question: "{query}"
        
        Guidelines:
        1. Stick strictly to the facts in the provided information.
        2. If specific information is not available to answer parts of the question, clearly say so.
        3. For date or time-based questions, provide the specific dates mentioned in the reports.
        4. If the question asks about parameters, certifications, or other specific details, highlight those in your answer.
        5. If comparing multiple reports, structure your answer to clearly show the comparison.
        6. Do not fabricate or infer information that isn't explicitly stated.
        7. Use a professional, factual tone appropriate for laboratory information.
        """
        
        # Generate response using Gemini
        try:
            with st.spinner("Generating response..."):
                response = gemini_model.generate_content(prompt)
                return response.text
        except Exception as e:
            st.error(f"Error generating response with Gemini: {e}")
            # Fallback response if Gemini fails
            return self.fallback_response(query, results)
    
    def fallback_response(self, query, results):
        """Generate a fallback response when Gemini fails"""
        response = "Based on the laboratory reports, I found:\n\n"
        
        for i, result in enumerate(results):
            if 'report' in result:
                report = result['report']
                response += f"Report {report.get('report_id', 'Unknown')}"
                
                if 'client_name' in report:
                    response += f" for {report['client_name']}"
                
                if 'report_date' in report:
                    response += f" (dated {report['report_date']})"
                
                response += "\n"
                
                # Add other relevant information
                if 'sample_id' in report:
                    response += f"- Sample ID: {report['sample_id']}\n"
                
                if 'analysis_start_date' in report and 'analysis_end_date' in report:
                    response += f"- Analysis period: {report['analysis_start_date']} to {report['analysis_end_date']}\n"
                
                if 'certifications' in report:
                    try:
                        certs = json.loads(report['certifications'])
                        if isinstance(certs, list):
                            response += f"- Certifications: {', '.join(certs)}\n"
                    except:
                        if isinstance(report['certifications'], str):
                            response += f"- Certifications: {report['certifications']}\n"
            
            if 'chunk' in result:
                chunk = result['chunk']
                # Take just the first 150 characters of the chunk for brevity
                chunk_preview = chunk.get('content', '')[:150] + "..."
                response += f"- Related content: {chunk_preview}\n"
            
            response += "\n"
        
        return response

# Initialize session state for history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Streamlit app
def main():
    st.set_page_config(
        page_title="GraphRAG Laboratory Reports Chatbot",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 GraphRAG Laboratory Reports Chatbot")
    st.markdown("""
    This application uses a Graph-based Retrieval Augmented Generation (GraphRAG) system to answer questions about laboratory reports.
    The system uses Neo4j for storing the data, SentenceTransformer for creating embeddings, and Google's Gemini for generating responses.
    """)
    
    # Initialize or access the GraphRAG system
    if 'graphrag' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.graphrag = GraphRAGSystem()
            st.success("System initialized!")
            
            # Check if Neo4j database already has data when starting up
            if st.session_state.graphrag.check_graph_populated():
                st.session_state.system_initialized = True
                st.success("Found existing data in Neo4j database! System is ready to use.")
    
    # Initialize query input state if not present
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""
    
    # Function to handle query submission
    def handle_submit():
        if st.session_state.query_input:
            query = st.session_state.query_input
            
            if not st.session_state.system_initialized:
                st.error("Please load data and build the graph first!")
            else:
                # Process query
                with st.spinner("Processing your query..."):
                    results = st.session_state.graphrag.query_graph(query)
                    response = st.session_state.graphrag.generate_response(query, results)
                
                # Add to history
                st.session_state.conversation_history.append((query, response))
            
            # Clear the input by setting it to empty string
            st.session_state.query_input = ""
    
    # Sidebar with system controls
    with st.sidebar:
        st.header("System Controls")
        
        # Only show the "Load Data" button if system is not initialized
        if not st.session_state.system_initialized:
            if st.button("Load Data & Build Graph"):
                with st.spinner("Loading data and building graph..."):
                    data = st.session_state.graphrag.load_data()
                    st.session_state.graphrag.build_knowledge_graph(data)
                    st.session_state.system_initialized = True
                    st.success("Knowledge graph built successfully!")
        else:
            st.success("System is ready to use!")
            
            # Add a button for forced reload if needed
            if st.button("Force Reload Data", help="Use this only if you need to refresh the data"):
                with st.spinner("Reloading data and rebuilding graph..."):
                    data = st.session_state.graphrag.load_data()
                    st.session_state.graphrag.build_knowledge_graph(data)
                    st.success("Knowledge graph rebuilt successfully!")
        
        st.header("Sample Queries")
        st.markdown("""
        Try these sample queries:
        - Show me reports for client ABC Farms
        - What are the results for sample ID S12345?
        - Were there any tests conducted between January 1, 2022 and March 15, 2022?
        - Show me reports certified under ISO 9001
        - What was the pH level in the most recent water quality test?
        """)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.conversation_history):
            st.markdown(f"**You:** {query}")
            st.markdown(f"**Chatbot:** {response}")
            st.divider()
        
        # Input for new query - using on_change to reset value
        query = st.text_input(
            "Ask a question about the laboratory reports:", 
            key="query_input",
            on_change=handle_submit
        )
        
        if st.button("Submit", key="submit_button"):
            handle_submit()
            
        # Show latest response if available
        if st.session_state.conversation_history:
            latest_query, latest_response = st.session_state.conversation_history[-1]
    
    with col2:
        # Display raw results for debugging
        if st.session_state.conversation_history:
            st.header("System Information")
            last_query = st.session_state.conversation_history[-1][0]
            
            with st.expander("View Raw Results"):
                if st.session_state.system_initialized:
                    results = st.session_state.graphrag.query_graph(last_query)
                    st.json(results)

if __name__ == "__main__":
    main()