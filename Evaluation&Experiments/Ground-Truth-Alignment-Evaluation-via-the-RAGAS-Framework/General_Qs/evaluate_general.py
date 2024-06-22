import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from ragas.metrics import (
    faithfulness,
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

def get_context(file_path="Context.txt"):
    """
    Load context data from a file and return a dictionary mapping queries to contexts.
    """
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            clean_line = line.strip()
            if clean_line:
                data = json.loads(clean_line)
                data_list.append(data)
    return {item['query']: [item['context'].split("Contextual Information:", 1)[1].split("Question:")[0].replace("\n", "")] for item in data_list}

def get_ground_truth(file_path="Ground_Truth.txt"):
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
    return {item['query']: item['Answer'] for item in data_list}

def get_second_ground_truth(file_path="Second_Ground_Truth.txt"):
    """
    Load secondary ground truth data from a file and return a dictionary mapping queries to ground truths.
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
    return {item['query']: item['Answer'] for item in data_list}

def extract_gpt4_answers(file_name="answers_gpt4.json"):
    """
    Extract GPT-4 answers from a JSON file and return as a list of dictionaries.
    """
    data_list = []
    try:
        with open(file_name, 'r') as file:
            content = file.read()
        corrected_content = re.sub(r'}\s*{', '},{', content)
        corrected_json = f'[{corrected_content}]'
        data = json.loads(corrected_json)
        data_list = [item for item in data]
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error extracting GPT-4 answers: {e}")
    return data_list

def build_dataset():
    """
    Build the dataset from extracted GPT-4 answers and context/ground truth data.
    """
    data_list = extract_gpt4_answers()
    queries, responses, contexts, ground_truths = [], [], [], []
    context_list = get_context()
    ground_truth_data = get_ground_truth()
    second_ground_data = get_second_ground_truth()

    for answer in data_list:
        if "Who wrote the NGSEC White Paper on Polymorphic Shellcodes vs. Application IDSs?" not in answer['query']:
            query = answer['query'] + '\n'
            context = context_list.get(query)
            if context:
                contexts.append(context)
                queries.append(query.strip())
                responses.append(answer["answer"])
                ground_truths.append(ground_truth_data.get(query, second_ground_data.get(query)))

    with open('comparison_queries.txt', 'a') as f:
        for query in queries:
            f.write(query)
            f.write("\n")

    df = pd.DataFrame({
        'question': queries,
        'answer': responses,
        'contexts': contexts,
        'ground_truth': ground_truths
    })

    dataset = Dataset.from_pandas(df)
    return DatasetDict({'eval': dataset})

# Initialize dataset
dataset_dict = build_dataset()

# Define evaluation function
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

# Evaluate and save results for the dataset
evaluate_and_save_results(dataset_dict, "RESULTS_GPT4.txt")

print('OK')
