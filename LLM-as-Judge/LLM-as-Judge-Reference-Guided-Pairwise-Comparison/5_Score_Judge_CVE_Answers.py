import os
import openai
import json
from tqdm import tqdm 
import json
import re
import time 

def configure():
  openai.api_key = os.getenv("AZURE_OPENAI_KEY")
  openai.api_base =  os.getenv("AZURE_OPENAI_ENDPOINT") 
  openai.api_type = '###'
  openai.api_version = "###"


def extract_questions():
    questions = []
    with open(r"question_test.txt", 'r', encoding='utf-8') as file:
        for line in file:
            questions.append(line.strip())  # Also consider stripping newline characters
    return questions


def extract_gpt4_answers():
    print("Extracting GPT4 answers ...")
    file_name = "GPT4_CVE_ANSWERS.txt"

    try:
        with open(file_name, 'r') as file:
            content = file.read()

        corrected_content = re.sub(r'}\s*{', '},{', content)
        corrected_json = f'[{corrected_content}]'
        data = json.loads(corrected_json)
        gpt4_dict = {}
        for item in data:
            gpt4_dict[item['query']] = item['answer']
    except json.JSONDecodeError as e:
        print("Errore durante la deserializzazione: ", e)
    except FileNotFoundError:
        print(f"Il file {file_name} non è stato trovato.")
    return gpt4_dict


def get_MoRSE_answers():
    file_path = "MoRSE_CVE.txt"

    # Lista per memorizzare i dizionari
    data_list = []

    # Leggi ogni riga del file di testo
    with open(file_path, "r") as file:
        for line in file:
            # Rimuovi eventuali spazi bianchi in eccesso e nuove righe
            line = line.strip()
            # Ignora le righe vuote
            if not line:
                continue
            # Converti la riga in un dizionario
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Errore nella riga JSON: {line}")
    # Dizionario per memorizzare le query e le risposte complete
    query_responses = {}
    queries = []
    # Ora data_list contiene tutti i dizionari letti dal file
    for answer in data_list:
        query = answer['query'].strip()
        response = answer["Answer"]
        if "[/INST]" in response:
            response = response.split("[/INST]", 1)[1]
        complete_answer = response
        queries.append(query)
        query_responses[query] = complete_answer
    
    return query_responses

def evalaute(question, ground_truth, answer):
    deployment_name='####' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    # Send a completion call to generate an answer
    print(f'[!] Answering question: {question}')
    #system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\."""
    system_prompt = """###Task Description:
        A Question, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
        3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
        4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
        ###Score Rubrics:
        [Is the response correct, accurate, and factual based on the reference answer?]
        Score 1: The response is completely incorrect, inaccurate, and/or not factual.
        Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
        Score 3: The response is somewhat correct, accurate, and/or factual.
        Score 4: The response is mostly correct, accurate, and factual.
        Score 5: The response is completely correct, accurate, and factual."""
    #prompt_template = f"[User Question]\n{question}\n\n[The Start of Reference Answer]\n{ground_truth}\n[The End of Reference Answer]\n\n[The Start of Assistant A's Answer]\n{answer_GPT}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_MoRSE}\n[The End of Assistant B's Answer]"

    prompt_template = f"###The Question: {question} ###Response to evaluate: {answer} ###Reference Answer (Score 5):{ground_truth}"
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template}
        ]
    print("[!] Sending Prompt ...")
    response = openai.ChatCompletion.create(deployment_id=deployment_name, messages=messages)
    answer = response['choices'][0]['message']['content']
    print(f"[!!!] Answer : {answer}")
    file_name = "MoRSE_CVE_VERDICT.json"
    dictionary = {"query": question, "answer":answer}
    with open(file_name, 'a') as file:
        json.dump(dictionary, file)
    print("[+] Answer Saved")
    
def set_ground_truth():
    file_path = "CVE_GROUND_TRUTH.txt"
    # Lista per memorizzare i dizionari
    data_list = []
    # Leggi ogni riga del file di testo
    with open(file_path, "r") as file:
        for line in file:
            # Rimuovi eventuali spazi bianchi in eccesso e nuove righe
            line = line.strip()
            # Ignora le righe vuote
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Errore nella riga JSON: {line}")
                print(e)
    data_dict = {item['CVE']: item['text'] for item in data_list}
    #print(data_dict)
    return data_dict


def start():
    for query, answer_MoRSE in MoRSE_results.items():
        count += 1
        try:
            print(f"[!] Question {str(count)} out of 52")
            print(f"[!] Processing query -> {query}")
            ground_truth = GROUND_TRUTH[query]
            evalaute(query, ground_truth, answer_MoRSE)
        except Exception as e:
            print(e)


def process_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Splitting the text at '---' to separate different vulnerabilities
    sections = content.split('---')

    cve_dict = {}

    for section in sections:
        # Finding the CVE number using regular expressions
        import re
        match = re.search(r'(CVE-\d{4}-\d+)', section)
        if match:
            cve_id = match.group(1)
            cve_dict[cve_id] = section.strip()

    return cve_dict

def perplexity():
    file_path = "PERPLEXITY_CVE_ANSWERS.txt"
    perplexity_dict = process_text_file(file_path)
    return perplexity_dict

MoRSE_results = get_MoRSE_answers()
GPT_results = extract_gpt4_answers()
PERPLEXITY_results = perplexity()
GROUND_TRUTH = set_ground_truth()

if __name__ == "__main__":
    print("[!] Configuring ...")
    configure()
    print("[!] Start Analysis ...")
    start()

