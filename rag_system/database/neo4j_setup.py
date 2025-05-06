from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_DIMENSION, VECTOR_INDEX_NAME

def get_neo4j_driver():
    """Create and return a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def init_neo4j_database():
    """Initialize the Neo4j database with necessary constraints and indexes."""
    with get_neo4j_driver().session() as session:
        # Create constraints for unique IDs
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Metric) REQUIRE e.name IS UNIQUE")
        
        # Create vector index for embeddings
        try:
            session.run(f"""
            CALL db.index.vector.createNodeIndex(
              '{VECTOR_INDEX_NAME}',
              'Chunk',
              'embedding',
              {EMBEDDING_DIMENSION},
              'cosine'
            )
            """)
            print(f"✓ Vector index '{VECTOR_INDEX_NAME}' created successfully")
        except Exception as e:
            # Index might already exist
            if "already exists" in str(e):
                print(f"✓ Vector index '{VECTOR_INDEX_NAME}' already exists")
            else:
                print(f"✗ Error creating vector index: {e}")
        
        print("✓ Neo4j database initialized with constraints and vector index")

def test_connection():
    """Test the connection to the Neo4j database."""
    try:
        with get_neo4j_driver().session() as session:
            result = session.run("CALL db.info()")
            info = result.single()
            print("✓ Successfully connected to Neo4j database")
            return True
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        return False

def drop_all_data():
    """WARNING: Drop all data from the database. Use with caution!"""
    if input("Are you sure you want to delete all data? Type 'YES' to confirm: ") != "YES":
        print("Operation cancelled")
        return
    
    with get_neo4j_driver().session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("✓ All data has been deleted from the database")

def get_database_statistics():
    """Return statistics about the nodes and relationships in the database."""
    with get_neo4j_driver().session() as session:
        node_counts = session.run("""
        MATCH (n)
        RETURN labels(n) AS label, count(*) AS count
        ORDER BY count DESC
        """)
        
        rel_counts = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
        """)
        
        print("Database Statistics:")
        print("-------------------")
        print("Node counts:")
        for record in node_counts:
            print(f"  {record['label']}: {record['count']}")
        
        print("\nRelationship counts:")
        for record in rel_counts:
            print(f"  {record['type']}: {record['count']}")