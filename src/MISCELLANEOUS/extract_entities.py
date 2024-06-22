import os
import re
import warnings
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain.document_loaders import AsyncHtmlLoader, PyPDFLoader, TextLoader, GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from haystack.nodes import EntityExtractor
from timeout_decorator import timeout, TimeoutError

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to get HuggingFace BGE embeddings
def get_embeddings(model_name="BAAI/bge-large-en", device="cuda"):
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={'normalize_embeddings': True}
    )

# Set GPU 4 as the default device (if needed)
# torch.cuda.set_device(4)

# Verify that GPU 4 is now the default device
print(torch.cuda.current_device())

# Initialize Entity Extractor
entity_extractor = EntityExtractor(model_name_or_path="dslim/bert-large-NER", use_gpu=True, devices=[torch.device('cuda:4'), torch.device('cuda:5')])

# Function to load and merge FAISS databases
def load_and_merge_faiss(db_prefix, num_chunks, embeddings):
    db = FAISS.load_local(f"{db_prefix}_chunk_1", embeddings)
    for i in range(2, num_chunks + 1):
        chunk_db = FAISS.load_local(f"{db_prefix}_chunk_{i}", embeddings)
        db.merge_from(chunk_db)
    return db

# Function to split text into chunks
def split_text(text_store):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function=len, is_separator_regex=False)
    text_contents = []
    for key, value in tqdm(text_store.items()):
        content = value.page_content
        splits = text_splitter.split_text(content)
        text_contents += splits
    return text_contents

# Load and merge text databases
hf_bge_embeddings = get_embeddings()
text_db = load_and_merge_faiss("text_chunk", 5, hf_bge_embeddings)
text_contents = split_text(text_db.docstore._dict)

# Load and merge Metasploit databases
meta_db = load_and_merge_faiss("metasploit_chunk", 5, hf_bge_embeddings)
metasploit_contents = split_text(meta_db.docstore._dict)

# Load and merge paper databases
paper_db = load_and_merge_faiss("paper_chunk", 3, get_embeddings("BAAI/bge-large-en-v1.5"))
paper_contents = split_text(paper_db.docstore._dict)

# Combine all contents
contents = paper_contents + metasploit_contents + text_contents
print("LEN CHUNKS: ", len(contents))
print("LEN PAPERS: ", len(paper_contents))
print("METASPLOIT LEN: ", len(metasploit_contents))
print("TEXT LEN", len(text_contents))

# Function to get the language model and tokenizer
def get_model():
    max_memory_mapping = {4: "5GB", 5: "5GB"}
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=True,
        max_memory=max_memory_mapping
    )
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side='left')
    streamer = TextStreamer(tokenizer)
    return model, tokenizer, streamer

model, tokenizer, streamer = get_model()

# Extract entities and generate descriptions
f = open("ENTITIES_DATABASE.txt", "a")
entities = []
access = False
count = 0

for doc in tqdm(contents):
    if "plications Conference (ACSAC 2007). IEEE, 2007, pp. 501–514." in doc:
        access = True
        print("Ok Let's go ...")
    if access and count <= 61257:
        count += 1
        try:
            entities_extracted = entity_extractor.extract(text=doc)
            for extracted in entities_extracted:
                entity_name = extracted["word"]
                if entity_name not in entities:
                    entities.append(entity_name)
                    print("\nEntity Extracted : ", entity_name)
                    print("\nContent: ", doc)
                    prompt = f"[INST] Please provide a detailed description of who or what {entity_name} is, in the context of the following information: {doc}[/INST]"
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:4")
                    generated_ids = model.generate(**inputs, max_length=500, temperature=0.1)
                    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    print("+++++++++++++++++++++++++++++++++++++++++++++++")
                    print(outputs[0])
                    print("Saving Result")
                    f.write(f"\n\n\n\n{outputs[0]}")
        except Exception as e:
            print(e)
