import pandas as pd
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from tqdm import tqdm
import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
import re

# Repository path
repo_path = "exploit_db_repo/"

# Initialize HuggingFace BGE embeddings
hf_bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                                             model_kwargs={"device": "cuda"},
                                             encode_kwargs={'normalize_embeddings': True})

# Function to load documents
def loader(suffix):
    python_loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[suffix],
        parser=LanguageParser(language=Language.PYTHON),
    )
    python_documents = python_loader.load()
    return python_documents

# Load documents with specified suffixes
documents = []
languages = [".py", ".txt", ".c", ".cpp", ".rb", ".java", ".html", ".sh", ".pl", ".php", ".js", ".md"]
for language in tqdm(languages):
    docs = loader(language)
    for doc in docs:
        documents.append((doc.metadata, doc.page_content))

# Initialize a list for DataFrame data
data = []

# Use a regex to find CVEs in each document
pattern_cve = re.compile(r'\bCVE-\d{4}-\d{4,7}\b')
for metadata, doc in documents:
    cve_found = pattern_cve.findall(doc)

    # Add row to DataFrame only if CVEs are found and not already present
    if cve_found:
        for cve in cve_found:
            if cve not in [item[1] for item in data]:
                data.append((metadata, cve, doc))

# Create a DataFrame
df = pd.DataFrame(data, columns=['metadata', 'cve', 'document'])

# Display the DataFrame
print(df)
