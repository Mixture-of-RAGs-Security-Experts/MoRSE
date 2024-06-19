from flask import Flask, render_template, request, jsonify, Response
import requests
import logging
import time
from threading import Event
import sys
import pickle
import streamlit as st
import redis
from server_client import text
import json
from tailer import tail
from data_retrieval_system import DataRetrievalSystem
from malware_retriever import MalwareRetriever, MalwareFileRetriever
import pandas as pd

class WebSocketClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.setup_logging()
        print("\n[!] Setting Retrieval System ...\n")
        self.emergency_text = None
        self.query = None
        self.trial_emergency = 0
        self.attempts = 1
        self.cuda_memory_error = False
        self.local_url = 'http://146.48.62.103:5000'
        self.data_retrieval_system = self.set_retrieval_system()

    def set_retrieval_system(self):
        """Initialize the DataRetrievalSystem."""
        try:
            start_time = time.time()
            data_retrieval_system = DataRetrievalSystem()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n[!] Time Required to build Retrieval System -> {elapsed_time} Seconds")
            return data_retrieval_system
        except KeyboardInterrupt:
            print("\n[!] Closing Retrieval System Building ...")
            sys.exit(1)

    def print_configurations(self):
        """Print current configurations for debugging."""
        print(f"\n[!] Emergency Text -> {self.emergency_text}")
        print(f"\n[!] Query -> {self.query}")
        print(f"\n[!] No. Trial Emergency -> {self.trial_emergency}")
        return

    def setup_logging(self):
        """Set up logging for the WebSocketClient."""
        logging.basicConfig(filename='websocket_client.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def raise_interrupt(self):
        """Send an interrupt signal to the server."""
        print("\n[+] SENDING INTERRUPT SIGNAL")
        try:
            response = requests.post(f'{self.server_url}/interrupt')
            if response.status_code == 200:
                self.logger.info(f"INTERRUPT request successful. Waiting for new_text events...")
            else:
                self.logger.error(f"Error in interrupt request. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"[!] raise_interrupt -> SERVER IS NOT REACHABLE: {conn_err}")
            self.logger.error(f"SERVER IS NOT REACHABLE: ERROR in raise_interrupt: {conn_err}")

    def is_server_reachable(self):
        """Check if the server is reachable."""
        try:
            response = requests.post(f'{self.server_url}/is_reachable')
            if response.status_code == 200:
                self.logger.info(f"REACHABLE request successful. Waiting for new_text events...")
                return response.status_code == 200
            else:
                self.logger.error(f"Error in reachable request. Status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as conn_err:
            print(f"[!] SERVER IS NOT REACHABLE: {conn_err}")
            self.logger.error(f"SERVER IS NOT REACHABLE: ERROR in is_server_reachable: {conn_err}")
            return False

    def sending_prompt(self, prompt):
        """Send a prompt to the server."""
        logging.info("[+] POST Request sending ...")
        response = requests.post(f'{self.server_url}/generate_output', json={'prompt': prompt})
        print("[+] prompt sent ...")
        logging.info("[+] prompt sent ...")
        if response.status_code == 200:
            self.logger.info(f"POST request successful. Waiting for new_text events...")
        else:
            self.logger.error(f"Error in POST request. Status code: {response.status_code}")

    def set_prompt(self, query, no_gpu):
        """Set and send the prompt."""
        try:
            try:
                prompt, combined_text = self.data_retrieval_system.run(query, cpu=no_gpu)
            except KeyboardInterrupt:
                print("[-] Stopping Retrieval System Execution")
            self.emergency_text = combined_text
            self.query = query
            self.print_configurations()
            print("[+] Retrieval Completed. Sending prompt ...")
            return prompt
        except Exception as e:
            print(f"[-] An error occurred in set_prompt -> {e}")
            print(f"[-] set_prompt values: \nquery -> {query}, \nno_gpu -> {no_gpu}")

    def send_request(self, query, emergency_prompt=None):
        """Send a request to the server."""
        if query:
            if emergency_prompt is None:
                try:
                    if not self.is_server_reachable():
                        info = f"[-] CPU -> Server at {self.server_url} is not reachable."
                        self.logger.error(info)
                        print(info)
                        no_gpu = True
                    elif self.is_server_reachable():
                        info = f"[+] GPU -> Remote Server at {self.server_url} is reachable"
                        self.logger.info(info)
                        print(info)
                        no_gpu = False

                    try:
                        prompt = self.set_prompt(query, no_gpu)
                    except TypeError:
                        print(f"[-] TypeError in send_request: \nquery -> {query}, \nno_gpu -> {no_gpu}")
                        self.set_prompt(query, False)
                    return prompt
                except requests.exceptions.ConnectionError as conn_err:
                    print(f"[: (] Connection Error: {conn_err}")
                    self.logger.error(f"[-] Connection Error: {conn_err}")
                    return f"[-] Connection Error: {conn_err}"
                except KeyboardInterrupt:
                    self.raise_interrupt()
        elif emergency_prompt:
            emergency_info = "[-] Trying emergency prompt ..."
            print(emergency_info)
            self.logger.error(emergency_info)
            self.sending_prompt(emergency_prompt)

    def save_to_file(self, filename='web_socket_client.pkl'):
        """Save the WebSocketClient to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

@st.cache_resource
def load_websocket_client(filename='web_socket_client.pkl'):
    """Load the WebSocketClient from a file."""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

url = 'http://127.0.0.1:5000/receive_query'
server_url = 'http://146.48.62.238:6000'

@st.cache_resource
def get_retrieval_system(server_url):
    """Get the retrieval system."""
    return WebSocketClient(server_url)

def leggi_file_json(file_path):
    """Read data from a JSON file."""
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(e)

file_path = 'received_text.json'

def clear_file():
    """Clear the contents of the JSON file."""
    data = {"text": ""}
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def handle_cuda_error(web_socket_client, query):
    """Handle CUDA memory errors."""
    try:
        if web_socket_client.trial_emergency == 1:
            print("[!] EMERGENCY LEVEL -> 2")
            print(f"[+] Trying with Info Exfiltration ...")
            filtered_prompt = web_socket_client.data_retrieval_system.set_emergency_prompt(web_socket_client.emergency_text, web_socket_client.query, True)
            web_socket_client.attempts += 1
            print({"attempts:": web_socket_client.attempts, "trial_emergency": web_socket_client.trial_emergency, "text": web_socket_client.emergency_text[:500]})
            print(" ------------ FILTERED PROMPT BELOW -------------------------")
            print(filtered_prompt[:500])
            web_socket_client.raise_interrupt()
        elif web_socket_client.trial_emergency == 0 and web_socket_client.cuda_memory_error == False:
            print("[!] EMERGENCY LEVEL -> 1")
            print("[+] Trying with code Exfiltration ...")
            filtered_prompt = web_socket_client.data_retrieval_system.set_emergency_prompt(web_socket_client.emergency_text, web_socket_client.query, False)
            print({"attempts:": web_socket_client.attempts, "trial_emergency": web_socket_client.trial_emergency, "text": web_socket_client.emergency_text[:500]})
            print(" ------------ FILTERED PROMPT BELOW -------------------------")
            print(filtered_prompt[:500])
            web_socket_client.cuda_memory_error = True
            web_socket_client.trial_emergency = 1
            web_socket_client.raise_interrupt()
            print('[!] Interrupt SENT ...')
    except Exception as e:
        print(f"Exception Captured -> emergency in on_new_text: {e}")
        web_socket_client.logger.error(f"Exception Captured -> emergency in on_new_text: {e}")
    return filtered_prompt

def set_empty_text():
    """Set the text field in the JSON file to an empty string."""
    data = {"text": ""}
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def read_and_update(file_path, container):
    """Read and update the text content from the JSON file."""
    text = ""
    current = ""
    previous_global_variable = ""
    cuda_memory_error = False

    while '</s>' not in current:
        try:
            config_data = leggi_file_json(file_path)
            if config_data:
                current_global_variable = config_data.get('text', '')
                current = current_global_variable
                if current_global_variable != previous_global_variable:
                    previous_global_variable = current_global_variable
                    text += current_global_variable
                    container.markdown(text)
                    print(f'Valore globale attuale: {current_global_variable}')
                    if 'CUDA out of memory' in current_global_variable:
                        set_empty_text()
                        cuda_memory_error = True
                        break
        except Exception as e:
            print(e)
            pass
    return cuda_memory_error, text

@st.cache_resource
def set_malware_retriever():
    """Set the malware retriever."""
    malware_retriever = MalwareRetriever()
    malware_retriever_instance = malware_retriever.set_retriever()
    malware_dict = malware_retriever.malware_dict
    return malware_retriever_instance, malware_dict

@st.cache_resource
def set_malware_file_retriever():
    """Set the malware file retriever."""
    malware_file_retriever = MalwareFileRetriever()
    malware_file_dict = malware_file_retriever.load_malware_file_dict_from_json("MALWARE_FILE_CODE.json")
    loaded_file_retriever = malware_file_retriever.load_file_retriever()
    return loaded_file_retriever, malware_file_dict

def show_malware(query, malware_retriever_instance, malware_dict):
    """Show malware information based on the query."""
    start_time = time.time()
    try:
        for name in malware_retriever_instance.get_relevant_documents(query):
            st.sidebar.info(name.page_content)
            malware_list = malware_dict[name.page_content]
            file_names = [tupla[0] for tupla in malware_list]
            codes = [tupla[1] for tupla in malware_list]
            filtered_data = [s.rsplit('\\', 2)[-1] for s in file_names]
            data = {
                'file': filtered_data,
                'codes': codes
            }
            df = pd.DataFrame(data)
            st.sidebar.dataframe(
                df,
                column_config={"file": "Malware File", "codes": "File Code"},
                hide_index=True,
            )
            st.sidebar.divider()
            end_time = time.time()
        print(f"elapsed time : {end_time - start_time}")
    except Exception as e:
        print(e)

def retrieve_script(query, loaded_file_retriever, malware_file_dict):
    """Retrieve scripts related to the query."""
    scripts = []
    try:
        for name in loaded_file_retriever.get_relevant_documents(query):
            file = name.page_content
            code = malware_file_dict[file]
            scripts.append((file, code))
    except Exception as e:
        print(f"ERROR IN retrieve_script -> {e}")
    return scripts

def set_prompt_script(script, query):
    """Set the prompt for the script."""
    try:
        prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                    Answer the Question {query} using the following additional code.
                    Code Title:
                    {script[0]}
                    Code:
                    {script[1]} [/INST]"""
    except Exception as e:
        print(f"ERROR IN set_prompt_script -> {e}")
    return prompt

def main():
    """Main function to run the Streamlit app."""
    file_presence = False
    st.title("🧠 MoRSE - Mixture of RAG Security Experts")
    malware_retriever_instance, malware_dict = set_malware_retriever()
    loaded_file_retriever, malware_file_dict = set_malware_file_retriever()
    with st.spinner('Building Retrieval System ...'):
        try:
            start_time = time.time()
            print("[+] Try Loading System Retrieval")
            web_socket_client = load_websocket_client()
            if web_socket_client is None:
                web_socket_client = get_retrieval_system(server_url)
                print("[+] Saving File ...")
                web_socket_client.save_to_file()
                print("[+] File Saved ...")
            else:
                file_presence = True
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[+] System Loaded. Time Required: {elapsed_time}")
        except Exception as e:
            print(f'[!] Exception Loading pkl file -> {e}')
            web_socket_client = get_retrieval_system(server_url)
    st.info("Retrieval System is ready. ✅")
    container = st.empty()
    query = st.chat_input("Say something")
    set_empty_text()
    if st.button('Interrupt'):
        web_socket_client.raise_interrupt()
    if query:
        show_malware(query, malware_retriever_instance, malware_dict)
        with st.spinner('Running Retrieval System ...'):
            prompt = web_socket_client.send_request(query=query)
        params = {'query': prompt}
        response = requests.post(url, json=params)
        cuda_memory_error, text = read_and_update(file_path, container)
        if cuda_memory_error:
            cuda_memory_error = False
            print("[!] Handling CUDA ERROR ...")
            prompt = handle_cuda_error(web_socket_client, query)
            if prompt:
                params = {'query': prompt}
                response = requests.post(url, json=params)
                read_and_update(file_path, container)
            else:
                print("[-] NO PROMPT")
        container.empty()
        st.info(text)
        scripts = retrieve_script(query, loaded_file_retriever, malware_file_dict)
        if scripts:
            st.markdown(''':red[**Additional Scripts**]''')
            for script in scripts:
                prompt = set_prompt_script(script, query)
                if prompt:
                    clear_file()
                    params = {'query': prompt}
                    response = requests.post(url, json=params)
                    st.divider()
                    st.write(script[0])
                    st.code(script[1])
                    _, text = read_and_update(file_path, container)
                    st.info(text)
            container.empty()
        print(text)
        clear_file()

if __name__ == "__main__":
    main()
