from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
from database.neo4j_store import vector_search

def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def search_similar_chunks(query, top_k=5):
    model = get_embedding_model()
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Perform vector search in Neo4j
    results = vector_search(query_embedding, limit=top_k)
    
    return results

def format_search_results(results):
    formatted_results = ""
    
    for i, result in enumerate(results):
        formatted_results += f"Chunk {i+1} (Score: {result['score']:.4f}):\n"
        formatted_results += f"{result['text']}\n\n"
    
    return formatted_results