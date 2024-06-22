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
import re
from tqdm import tqdm

# Set environment variables
os.environ["OPENAI_API_KEY"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Initialize the OpenAI API client
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# Apply nest_asyncio to avoid runtime errors in Jupyter notebooks
nest_asyncio.apply()

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

def get_context():
    """
    Load context data from a file and return a dictionary mapping queries to contexts.
    """
    file_path = "MoRSE_Answer_MultiHop_Ground_Truth.txt"
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            clean_line = line.strip()
            if clean_line:
                data = json.loads(clean_line)
                data_list.append(data)
    return {item['query']: [item['context'].split("Infos:\n(", 1)[1]] for item in data_list}

def get_ground_truth(file_path="MoRSE_Answer_MultiHop_Ground_Truth.txt"):
    """
    Load ground truth data from a file and return a dictionary mapping queries to ground truths.
    """
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error in JSON line: {line}")
                    print(e)
    return {item['query']: item['ground_truth'] for item in data_list}

def build_dataset(file_path, context_list, ground_truth_data, second_ground_data=None):
    """
    Build a dataset from the provided file and context/ground truth data.
    """
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error in JSON line: {line}")

    queries, responses, contexts, ground_truths = [], [], [], []
    for answer in data_list:
        query = answer['query'] + '\n'
        context = context_list.get(query)
        if context:
            contexts.append(context)
            queries.append(query.strip())
            responses.append(answer["answer"])
            ground_truths.append(ground_truth_data.get(query, second_ground_data.get(query) if second_ground_data else None))

    df = pd.DataFrame({
        'question': queries,
        'answer': responses,
        'contexts': contexts,
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
metrics = [answer_relevancy, answer_similarity, answer_correctness]

# Load context and ground truth data
context_list = get_context()
ground_truth_data = get_ground_truth()
second_ground_data = get_ground_truth("Second_Ground_Truth.txt")

# Build datasets from different sources
dataset_dict_hacker = build_dataset("HACKERGPT_RESULTS.txt", context_list, ground_truth_data, second_ground_data)
dataset_dict_gemini = build_dataset("GEMINI_ANSWERS.txt", context_list, ground_truth_data, second_ground_data)
dataset_dict_mixtral = build_dataset("MULTIHOP_Answer_MIXTRAL.txt", context_list, ground_truth_data, second_ground_data)

# Evaluate each dataset and save results to respective files
def evaluate_and_save_results(dataset_dict, result_file):
    """
    Evaluate each entry in the dataset individually and save results to a file.
    """
    with open(result_file, "w") as f:
        for i in range(len(dataset_dict["eval"])):
            single_row_dataset = dataset_dict["eval"].select([i])
            result = evaluate(
                single_row_dataset, metrics=metrics, llm=azure_model, embeddings=embeddings
            )
            f.write(str(result) + "\n")
            print(result)

# Evaluate and save results for each dataset
evaluate_and_save_results(dataset_dict_hacker, "RESULTS_MULTIHOP_HACKERGPT.txt")
evaluate_and_save_results(dataset_dict_gemini, "RESULTS_MULTIHOP_GEMINI.txt")
evaluate_and_save_results(dataset_dict_mixtral, "RESULTS_MULTIHOP_MIXTRAL.txt")

print('OK')
