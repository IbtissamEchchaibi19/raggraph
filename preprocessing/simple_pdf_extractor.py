import os
import PyPDF2

# Fixed directory names
INPUT_DIR = "C:/Users/ibtis/OneDrive/Desktop/TestLab/searchablepdf"
OUTPUT_DIR = "C:/Users/ibtis/OneDrive/Desktop/ProccedData/NLP"

def extract_text_from_pdf(pdf_path):
    """Extract text from a searchable PDF file"""
    text = ""
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def process_pdfs():
    """Process all PDF files in the searchabledocs directory"""
    
    # Make sure the input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory '{INPUT_DIR}' not found. Creating it...")
        os.makedirs(INPUT_DIR)
        print(f"Please put your PDF files in the '{INPUT_DIR}' folder and run the script again.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Starting extraction...")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        
        # Create a folder with the PDF name (without extension)
        folder_name = os.path.splitext(pdf_file)[0]
        output_folder = os.path.join(OUTPUT_DIR, folder_name)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Extract text
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text:
            # Save to raw_text.txt file inside the subdirectory
            output_file = os.path.join(output_folder, "raw_text.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"Extracted text from {pdf_file} and saved to {output_folder}/raw_text.txt")
        else:
            print(f"Failed to extract text from {pdf_file}")

    print("\nExtraction complete!")
    print(f"Processed {len(pdf_files)} PDF files")
    print(f"Text files are saved in the '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    print("PDF Text Extractor")
    print("=================")
    process_pdfs()
