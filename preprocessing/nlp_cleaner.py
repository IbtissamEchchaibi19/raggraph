import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
NLP_DIRECTORY = "C:/Users/ibtis/OneDrive/Desktop/ProccedData/NLP"

def clean_with_gemini(text):
    """Use Gemini API to clean text and extract metadata"""
    # Configure Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
You are an expert in document parsing and scientific text cleaning.

I need you to clean the following lab report text and separate it into two distinct parts: the cleaned scientific content and the metadata.

=======================
STRICT CLEANING RULES:
=======================
1. REMOVE:
   - Arabic words, letters, or transliterations (e.g., "Al-", "Ash-", or Arabic script)
   - Gibberish or unreadable symbols
   - Page numbers, footers, headers, and formatting artifacts (like "Page X of Y")
   - Report IDs, reference numbers, dates, and barcode strings
   - Phone numbers, email addresses, physical addresses, website links
   - Logos, certifications (e.g., ISO 17025, TÜV), or quality seals
   - Special characters (e.g., excessive slashes, backslashes, underscores)
   - Redundant or irregular whitespace and line breaks

2. KEEP:
   - Exact values in scientific measurements (do not alter numbers, symbols like %, °C, etc.)
   - Test parameters, methods, results, units, and reference values
   - Section titles and test result structure (preferably in table format)
   - Scientific terminology (chemical names, compound names, etc.)

=======================
CRITICAL STRUCTURING:
=======================
- Clean the document into readable, logically formatted paragraphs
- Test results should be presented as structured tables or clearly labeled sections
- Group related results together under proper headings (e.g., "Microbiological Analysis", "Heavy Metals")

=======================
METADATA EXTRACTION:
=======================
Extract and list separately under a "metadata" field:
- Report date, issue date, sample collection and reception dates
- Report number or sample ID
- Client name, company, and address (if present)
- Contact person, phone, email (if present)
- Purchase order or request reference
- ISO/certification numbers or names
- Any barcode or alphanumeric sample identifier
DO NOT include metadata in the cleaned text section.

=======================
RETURN FORMAT:
=======================
Respond ONLY with a valid JSON object using this structure:
{{
  "cleaned_text": "<<<Clean scientific report content>>>",
  "metadata": {{
    "report_number": "...",
    "client_name": "...",
    "client_address": "...",
    "contact_email": "...",
    "sample_id": "...",
    "report_date": "...",
    "certifications": [...],
    ...
  }}
}}

Here is the text to clean and extract:

{text}
"""
    
    try:
        response = model.generate_content(prompt)
        result = response.text
        
        # Extract the JSON part of the response
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = result[start_idx:end_idx]
            json_data = json.loads(json_str)
            
            # Extract cleaned text and metadata
            cleaned_text = json_data.get("cleaned_text", "")
            metadata = json_data.get("metadata", {})
            
            return cleaned_text, metadata
        else:
            # Simple fallback if JSON parsing fails
            return text, {"error": "Failed to extract metadata"}
                
    except Exception as e:
        print(f"Error in Gemini API call: {str(e)}")
        return text, {"error": str(e)}

def process_file(file_path):
    """Process a single text file"""
    try:
        print(f"Processing: {file_path}")
        
        # Check if the file has already been processed
        cleaned_file_path = file_path.replace('.txt', '_cleaned.txt')
        metadata_file_path = file_path.replace('.txt', '_metadata.json')
        
        if os.path.exists(cleaned_file_path) and os.path.exists(metadata_file_path):
            print(f"✓ Already processed: {os.path.basename(file_path)}")
            return
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Clean the content using Gemini
        cleaned_text, metadata = clean_with_gemini(content)
        
        # Save the cleaned text
        with open(cleaned_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Save the metadata
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully processed: {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(file_path)}: {str(e)}")

def process_directory():
    """Process all text files in the directory structure"""
    print(f"Starting to process directory: {NLP_DIRECTORY}")
    
    # Ensure the directory exists
    if not os.path.exists(NLP_DIRECTORY):
        print(f"Error: Directory {NLP_DIRECTORY} does not exist.")
        return
    
    # Find all text files in the directory and subdirectories
    text_files = []
    for root, _, files in os.walk(NLP_DIRECTORY):
        for file in files:
            if file.endswith('.txt') and not file.endswith('_cleaned.txt') and not file.endswith('_metadata.json'):
                text_files.append(os.path.join(root, file))
    
    print(f"Found {len(text_files)} text files to process")
    
    if not text_files:
        print("No files to process.")
        return
    
    # Process files one by one
    processed_count = 0
    error_count = 0
    
    for file in text_files:
        try:
            process_file(file)
            processed_count += 1
        except Exception as e:
            error_count += 1
            print(f"✗ Unexpected error with {os.path.basename(file)}: {str(e)}")
        
        print(f"Progress: {processed_count + error_count}/{len(text_files)} files (Success: {processed_count}, Errors: {error_count})")
            
    print(f"Directory processing complete! Successfully processed {processed_count} files with {error_count} errors.")

# Main execution
if __name__ == "__main__":
    # Check if API key is available
    if not API_KEY:
        print("Error: Gemini API key not found. Please check your .env file.")
    else:
        process_directory()