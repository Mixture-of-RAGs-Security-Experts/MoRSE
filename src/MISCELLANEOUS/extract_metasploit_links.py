import os
import pandas as pd
import re
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
from langchain.document_loaders import AsyncHtmlLoader, PyPDFLoader, GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Clone repository path
repo_path = "METASPLOIT_DB/"

# Load HuggingFace BGE embeddings
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cuda"},
    encode_kwargs={'normalize_embeddings': True}
)

# Load documents from a specific file suffix
def loader(suffix):
    python_loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[suffix],
        parser=LanguageParser(),
    )
    python_documents = python_loader.load()
    return python_documents

# Load documents with specified languages
documents = []
languages = [".rb"]
for language in tqdm(languages):
    docs = loader(language)
    documents += docs

# Extract links from the loaded documents
total_links = []
for i, doc in enumerate(documents):
    print(f"Examining document: {i}")
    links = re.findall(r'https?://\S+', doc.page_content)
    for link in links:
        if "https://github.com/rapid7/metasploit-framework" not in link and "https://metasploit.com/download" not in link and "youtube" not in link:
            link = link.replace("'", "").replace("]", "").replace(",", "")
            if link not in total_links:
                total_links.append(link)
                print(link)

print(len(total_links))

# Analyze and load documents from the extracted links
total_REFERENCES = []
total_documents = []
for href in total_links:
    try:
        if "https" in href and "kaspersky" not in href and "author" not in href and ".tags" not in href and ".lat" not in href and "securelist.com/category/" not in href and ".ru" not in href:
            if href not in total_REFERENCES:
                try:
                    total_REFERENCES.append(href)
                    print("---------------------------------")
                    print("Analyzing ref: " + href)

                    @timeout(40)  # 40 seconds timeout, adjust as needed
                    def analyze_href(href):
                        if ".pdf" in href:
                            loader = PyPDFLoader(href, extract_images=True)
                            pages = loader.load()
                            for page in pages:
                                if page not in total_documents:
                                    total_documents.append(page)
                        else:
                            loader = AsyncHtmlLoader(href)
                            docs = loader.load()
                            html2text = Html2TextTransformer()
                            docs_transformed = html2text.transform_documents(docs)
                            for document in docs_transformed:
                                if document not in total_documents:
                                    total_documents.append(document)

                    analyze_href(href)
                    print("\nTotal Documents :")
                    print(len(total_documents))
                except TimeoutError:
                    print("Timeout exceeded for: " + href)
                except Exception as e:
                    print(f"Error analyzing ref {href}: {str(e)}")
    except Exception as e:
        print(e)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
splits = text_splitter.split_documents(total_documents)

# Initialize HuggingFace BGE embeddings
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Build and save FAISS database
print("\n[!!!] Building DATABASE")
faiss_db = FAISS.from_documents(splits, embeddings)
faiss_db.save_local("METASPLOIT_LINKS/links_db")
print("\n[!!!] DATABASE BUILT")
