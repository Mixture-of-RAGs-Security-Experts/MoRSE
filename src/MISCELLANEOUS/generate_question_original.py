import os
import pandas as pd
from haystack.nodes import PDFToTextConverter, PreProcessor, QuestionGenerator
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import QuestionGenerationPipeline
from haystack.utils import print_questions
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_metasploit_dataframe():
    """
    Load and preprocess the Metasploit dataframe.
    Rename columns and invert the dataframe.
    """
    file_path = "cybersecurity_dataframe.csv"
    df_loaded = pd.read_csv(file_path)
    df_loaded.rename(columns={'page_content': 'content'}, inplace=True)
    df_inverted = df_loaded.iloc[:, ::-1]
    return df_inverted

def generate_and_save_questions(file, output_dir):
    """
    Generate questions from the given file content and save them to a specified directory.
    """
    # Initialize a document store
    document_store = InMemoryDocumentStore(use_bm25=True)

    # Write the preprocessed document to the document store
    docs = [{"content": str(file)}]
    document_store.write_documents(docs)

    # Initialize the question generator and pipeline
    question_generator = QuestionGenerator()
    question_generation_pipeline = QuestionGenerationPipeline(question_generator)
    output_file_path = os.path.join(output_dir, "questions.txt")

    for idx, document in enumerate(document_store):
        generate_questions(question_generation_pipeline, output_file_path, idx, document)

    print(f"Questions generated and saved to {output_file_path}")

def generate_questions(question_generation_pipeline, output_file_path, idx, document):
    """
    Generate questions for a single document and save them to a file.
    """
    print(f"\n * Generating questions for document {idx}: {document.content[:100]}...\n")
    result = question_generation_pipeline.run(documents=[document])
    print_questions(result)
    
    # Save the generated questions to a text file
    print(f"Saving questions to {output_file_path}")
    with open(output_file_path, "a") as output_file:
        output_file.write(str(result) + "\n\n\n")

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200,
    length_function = len,
    is_separator_regex = False,
)

# Load the dataframe
df = load_metasploit_dataframe()
print(df)

# Directory to save the generated questions
output_directory = "Generated_Questions_Original/"
os.makedirs(output_directory, exist_ok=True)

access = 0
for index, row in df.iterrows():
    source = row['source']
    content = row['content']
    print(source)
    
    if access == 0 and "www.macexploit.com" in source:
        access = 1
    
    # Use the 'source' and 'content' variables in each iteration
    if access == 1:
        try:
            print(f"Source: {source}\n")
            subdirectory = os.path.join(output_directory, source)
            os.makedirs(subdirectory, exist_ok=True)
            splits = text_splitter.split_text(content)

            for split in splits:
                # Generate and save questions for each text split
                generate_and_save_questions(split, subdirectory)
        except Exception as e:
            print(e)
            pass
