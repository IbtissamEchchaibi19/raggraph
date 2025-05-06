from langchain_groq import ChatGroq
from config import GROQ_API_KEY
from retrieval.query_classifier import is_query_complex
from retrieval.vector_search import search_similar_chunks, format_search_results
from retrieval.cypher_generation import execute_generated_cypher, format_cypher_results

def generate_final_answer(question):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")
    is_complex = is_query_complex(question)
    if is_complex:
        cypher_query, cypher_results = execute_generated_cypher(question)
        context = format_cypher_results(cypher_results)
        retrieval_method = "graph"
    else:
        vector_results = search_similar_chunks(question, top_k=5)
        context = format_search_results(vector_results)
        retrieval_method = "vector"
    
    llm = ChatGroq(temperature=0, groq_api_key= GROQ_API_KEY, model_name="mixtral-8x7b-32768")
    prompt = f"""
    You are a question-answering assistant with access to relevant information.
    
    User Question: "{question}"
    
    Here is the relevant information retrieved from our knowledge base:
    {context}
    
    Based ONLY on the provided information, answer the user's question.
    If the information is insufficient to provide a complete answer, acknowledge what you know
    and what additional information would be needed.
    
    Provide a clear, concise, and accurate answer.
    """
    
    # Generate answer with LLaMA
    answer = llm.invoke(prompt).strip()
    
    # Return the answer and metadata
    return {
        "question": question,
        "answer": answer,
        "retrieval_method": retrieval_method,
        "is_complex_query": is_complex,
        "context_used": context
    }

def format_response(response_dict):
 
    formatted_response = f"Answer: {response_dict['answer']}\n\n"
    
    # Add debug information if requested (this could be controlled by a parameter)
    debug_mode = False
    if debug_mode:
        formatted_response += "---\nDebug Info:\n"
        formatted_response += f"Query classified as: {'Complex' if response_dict['is_complex_query'] else 'Simple'}\n"
        formatted_response += f"Retrieval method: {response_dict['retrieval_method']}\n"
    
    return formatted_response

def process_user_query(question, include_debug_info=False):
    try:
        # Generate the answer and metadata
        response_dict = generate_final_answer(question)
        
        # Format the response
        if include_debug_info:
            return f"Answer: {response_dict['answer']}\n\n" + \
                   f"Query type: {'Complex' if response_dict['is_complex_query'] else 'Simple'}\n" + \
                   f"Retrieval method: {response_dict['retrieval_method'].capitalize()}\n" + \
                   f"Context used: \n{response_dict['context_used']}"
        else:
            return response_dict['answer']
            
    except Exception as e:
        return f"Error processing query: {str(e)}"