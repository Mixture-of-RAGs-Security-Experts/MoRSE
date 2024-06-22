import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from ragas.metrics import (
    answer_relevancy,
    context_relevancy,
    context_precision,
    AnswerCorrectness,
    ContextRecall,
    AnswerSimilarity
)
from ragas import evaluate
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import nest_asyncio
import openai
from tqdm import tqdm

# Initialize OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = ""
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# Apply nest_asyncio to avoid runtime errors in Jupyter notebooks
nest_asyncio.apply()

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set other environment variables for Azure and HuggingFace
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Initialize Azure OpenAI model configurations
azure_configs = {
    "base_url": "",
    "model_deployment": "",
    "model_name": "gpt-4",
    "embedding_deployment": "",
    "embedding_name": "text-embedding-ada-002",
}

# Initialize Azure OpenAI model
azure_model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
)

# Initialize Azure OpenAI embeddings
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)

def get_ground_truth(file_path="cve_ground_truth.txt"):
    """
    Load ground truth data from a file and return a dictionary mapping CVEs to text.
    """
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error in JSON line: {line}")
                print(f"ERROR: {e}")
    return {item['CVE']: item['text'] for item in data_list}

def build_dataset(file_path, ground_truth_data):
    """
    Build a dataset from the provided file and ground truth data.
    """
    data_list = []
    with open(file_path, "r") as file:
        text = file.read()
        lines = text.split("}{")
        for line in lines:
            line = line.strip()
            line = "{" + line + "}"
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error in JSON line: {line}")

    queries, responses, ground_truths = [], [], []
    for answer in data_list:
        query = answer['query']
        try:
            ground_truths.append(ground_truth_data[query])
            queries.append(query)
            responses.append(answer["answer"])
        except KeyError as e:
            print(e)

    df = pd.DataFrame({
        'question': queries,
        'answer': responses,
        'ground_truth': ground_truths
    })

    dataset = Dataset.from_pandas(df)
    return DatasetDict({'eval': dataset})

def evaluate_dataset(dataset):
    """
    Evaluate the dataset using specified metrics and return the results.
    """
    result = evaluate(
        dataset["eval"],
        metrics=[
            answer_relevancy,
            context_relevancy,
            context_precision,
            AnswerCorrectness,
            ContextRecall,
            AnswerSimilarity
        ],
    )
    return result

# Initialize HuggingFace BGE embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cuda:4"},
    encode_kwargs={"normalize_embeddings": True},
)

# Define metrics for evaluation
answer_correctness = AnswerCorrectness(weights=[0.4, 0.6])
answer_similarity = AnswerSimilarity()
metrics = [answer_similarity, answer_correctness]

# Load ground truth data
ground_truth_data = get_ground_truth()

# Build dataset from GPT-4 answers
dataset_dict = build_dataset("Answer_CVE_GPT4.txt", ground_truth_data)

# Evaluate each entry in the dataset individually and save results to a file
with open("RESULTS_CVE_GPT4.txt", "w") as f:
    for i in range(len(dataset_dict["eval"])):
        time.sleep(5)
        single_row_dataset = dataset_dict["eval"].select([i])
        result = evaluate(
            single_row_dataset, metrics=metrics, llm=azure_model, embeddings=embeddings
        )
        f.write(str(result) + "\n")
        print(result)
