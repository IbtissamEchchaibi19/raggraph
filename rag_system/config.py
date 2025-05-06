import os

# Neo4j Configuration
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small but effective model from sentence-transformers
EMBEDDING_DIMENSION = 384  # Dimension of the all-MiniLM-L6-v2 embeddings

# Text Chunking Configuration
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity

# Vector Index Configuration
VECTOR_INDEX_NAME = "chunk_embeddings"

# Query Processing Configuration
QUERY_COMPLEXITY_THRESHOLD = 0.6  # Threshold for determining complex queries

# Groq API Configuration (for LLM calls)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

# Debug Mode (set to True to see additional information)
DEBUG_MODE = False