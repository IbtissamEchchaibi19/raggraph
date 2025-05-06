from langchain_groq import ChatGroq
from config import GROQ_API_KEY, QUERY_COMPLEXITY_THRESHOLD

def is_query_complex(query):
    complex_indicators = [
        "compare", "comparing", "comparison",
        "relationship", "related to", "connected",
        "between", "and", "versus", "vs",
        "highest", "lowest", "top", "best", "worst",
        "average", "mean", "median", "sum", "total",
        "before", "after", "during", "when",
        "how many", "count", "number of",
        "which author", "which model", "which topic"
    ]
    
    query_lower = query.lower()
    
    # Check for complex indicators
    for indicator in complex_indicators:
        if indicator in query_lower:
            return True
    
    # If no complex indicators found, use LLM-based classification for more nuanced decisions
    if GROQ_API_KEY:
        return llm_classify_query_complexity(query)
    else:
        # If no API key, default to considering it simple
        return False

def llm_classify_query_complexity(query):
    try:
        llm =  ChatGroq(temperature=0, groq_api_key= GROQ_API_KEY, model_name="mixtral-8x7b-32768")
        
        prompt = f"""
        Determine if the following query requires graph-based reasoning or simple semantic similarity search.
        
        Query: "{query}"
        
        If this query involves relationships between entities, requires filtering or aggregation, 
        asks for comparisons or rankings, or needs temporal reasoning, classify it as COMPLEX.
        
        If this query is simply looking for information on a single topic or concept that can be 
        retrieved based on semantic similarity, classify it as SIMPLE.
        
        Return only one word: either "COMPLEX" or "SIMPLE".
        """
        
        result = llm.invoke(prompt).strip().upper()
        
        return "COMPLEX" in result
    except Exception as e:
        print(f"Error in LLM query classification: {e}")
        # Default to simple if there's an error
        return False