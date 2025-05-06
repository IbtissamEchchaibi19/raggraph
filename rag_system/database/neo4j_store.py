import uuid
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, VECTOR_INDEX_NAME
from database.neo4j_setup import get_neo4j_driver

def store_text_chunk(text, embedding, metadata=None):
    """
    Store a text chunk with its embedding and metadata in Neo4j.
    
    Args:
        text (str): The text content of the chunk
        embedding (list): The vector embedding of the text
        metadata (dict): Additional metadata like source, created_at, etc.
        
    Returns:
        str: The unique ID of the stored chunk
    """
    chunk_id = str(uuid.uuid4())
    
    with get_neo4j_driver().session() as session:
        # Create chunk node with text content and embedding
        session.run(
            """
            CREATE (c:Chunk {
                id: $id, 
                text: $text, 
                embedding: $embedding
            })
            """,
            id=chunk_id,
            text=text,
            embedding=embedding
        )
        
        # Add metadata if provided
        if metadata and isinstance(metadata, dict):
            properties = []
            params = {"chunk_id": chunk_id}
            
            for key, value in metadata.items():
                if value is not None:
                    properties.append(f"c.{key} = ${key}")
                    params[key] = value
            
            if properties:
                property_str = ", ".join(properties)
                query = f"""
                MATCH (c:Chunk {{id: $chunk_id}})
                SET {property_str}
                """
                session.run(query, **params)
    
    return chunk_id

def link_chunk_to_author(chunk_id, author_name):
    """
    Create a relationship between a chunk and an author.
    
    Args:
        chunk_id (str): The ID of the chunk
        author_name (str): The name of the author
    """
    with get_neo4j_driver().session() as session:
        session.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (a:Author {name: $author_name})
            MERGE (c)-[:HAS_AUTHOR]->(a)
            """,
            chunk_id=chunk_id,
            author_name=author_name
        )

def link_chunk_to_topic(chunk_id, topic_name):
    """
    Create a relationship between a chunk and a topic.
    
    Args:
        chunk_id (str): The ID of the chunk
        topic_name (str): The name of the topic
    """
    with get_neo4j_driver().session() as session:
        session.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (t:Topic {name: $topic_name})
            MERGE (c)-[:HAS_TOPIC]->(t)
            """,
            chunk_id=chunk_id,
            topic_name=topic_name
        )

def store_model_evaluation(model_name, metric_name, score):
    """
    Store model evaluation data in the graph.
    
    Args:
        model_name (str): The name of the model
        metric_name (str): The name of the evaluation metric
        score (float): The evaluation score
    """
    with get_neo4j_driver().session() as session:
        session.run(
            """
            MERGE (m:Model {name: $model_name})
            MERGE (e:Metric {name: $metric_name})
            MERGE (m)-[:EVALUATED_ON]->(e)
            MERGE (s:Score {value: $score})
            MERGE (m)-[:HAS_SCORE]->(s)
            MERGE (s)-[:FOR_METRIC]->(e)
            """,
            model_name=model_name,
            metric_name=metric_name,
            score=score
        )

def vector_search(query_embedding, limit=5):
    """
    Perform vector similarity search in Neo4j.
    
    Args:
        query_embedding (list): The query embedding vector
        limit (int): Maximum number of results to return
        
    Returns:
        list: Matching chunks with their similarity scores
    """
    with get_neo4j_driver().session() as session:
        result = session.run(
            f"""
            CALL db.index.vector.queryNodes(
              '{VECTOR_INDEX_NAME}',
              $limit,
              $query_embedding
            ) YIELD node, score
            RETURN node.id AS id, node.text AS text, node.source AS source, score
            """,
            limit=limit,
            query_embedding=query_embedding
        )
        
        return [{"id": record["id"], 
                 "text": record["text"], 
                 "source": record["source"],
                 "score": record["score"]} 
                for record in result]

def execute_cypher_query(cypher_query, parameters=None):
    """
    Execute a Cypher query in Neo4j.
    
    Args:
        cypher_query (str): The Cypher query to execute
        parameters (dict): Parameters for the query
        
    Returns:
        list: Results of the query as a list of dictionaries
    """
    if parameters is None:
        parameters = {}
        
    with get_neo4j_driver().session() as session:
        result = session.run(cypher_query, **parameters)
        return [dict(record) for record in result]