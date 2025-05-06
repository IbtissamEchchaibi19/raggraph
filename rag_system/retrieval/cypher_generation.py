from langchain_groq import ChatGroq
from config import GROQ_API_KEY
from database.neo4j_store import execute_cypher_query

def generate_cypher_query(question):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")
    
    llm = ChatGroq(temperature=0, groq_api_key= GROQ_API_KEY, model_name="mixtral-8x7b-32768")
    
    # Schema information to help the LLM generate correct Cypher
    schema_info = """
    Neo4j Graph Schema:
    - Nodes:
        - (Chunk): Text chunks with properties {id, text, embedding, source, created_at}
        - (Author): Author nodes with properties {name}
        - (Topic): Topic nodes with properties {name}
        - (Model): ML model nodes with properties {name}
        - (Metric): Evaluation metric nodes with properties {name}
        - (Score): Score nodes with properties {value}
    
    - Relationships:
        - (Chunk)-[:HAS_AUTHOR]->(Author)
        - (Chunk)-[:HAS_TOPIC]->(Topic)
        - (Model)-[:EVALUATED_ON]->(Metric)
        - (Model)-[:HAS_SCORE]->(Score)
        - (Score)-[:FOR_METRIC]->(Metric)
    """
    
    prompt = f"""
    {schema_info}
    
    Task: Generate a Cypher query for Neo4j that answers the following question:
    "{question}"
    
    The query should be syntactically correct and optimized.
    Return ONLY the Cypher query without any explanation or markdown formatting.
    """
    
    # Generate Cypher query using LLaMA
    cypher_query = llm.invoke(prompt).strip()
    
    # Clean up the response to ensure it's just the Cypher query
    # Sometimes models include backticks or comments
    cypher_query = cypher_query.replace('```cypher', '').replace('```', '').strip()
    
    return cypher_query

def execute_generated_cypher(question):
    cypher_query = generate_cypher_query(question)
    
    # Execute the query
    results = execute_cypher_query(cypher_query)
    
    return cypher_query, results

def format_cypher_results(results):
    if not results:
        return "No results found."
    
    formatted_results = "Query Results:\n"
    
    # Extract keys from the first result
    keys = results[0].keys()
    
    # Format each result row
    for i, result in enumerate(results):
        formatted_results += f"Result {i+1}:\n"
        for key in keys:
            formatted_results += f"  {key}: {result[key]}\n"
        formatted_results += "\n"
    
    return formatted_results