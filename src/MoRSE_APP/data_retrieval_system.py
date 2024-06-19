import logging
import time
import torch
from haystack.nodes import EntityExtractor
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

from mitre_retriever import MalwareAnalyzer
from exploit_retriever import CodeRetrievalSystem
from entity_retriever import EntityRetrievalSystem
from question_retrieval_system import QuestionRetrievalSystem
from unstructured_rag import CybersecurityAssistant

class DataRetrievalSystem:
    def __init__(self):
        # Setting up logging
        logging.basicConfig(
            filename="Cache_Main_Log.txt",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.main_logger = logging.getLogger("MainLogger")
        self.model, self.tokenizer, self.streamer = self.configure_model()

        # Creating instances of other classes
        self.code_retrieval_system = CodeRetrievalSystem()
        self.question_retrieval_system = QuestionRetrievalSystem()
        self.entity_retrieval_system = EntityRetrievalSystem()
        self.cybersecurity_assistant = CybersecurityAssistant()
        self.malware_analyzer = MalwareAnalyzer()

        # Cache to store the results of previous queries
        self.query_cache = {}

    def configure_model(self):
        """Configure and return the model, tokenizer, and streamer."""
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
        tokenizer.padding_side = "left"
        streamer = TextStreamer(tokenizer)
        return model, tokenizer, streamer

    def run(self, query, cpu=False):
        """Run the data retrieval process for a given query."""
        entity_extractor = EntityExtractor(model_name_or_path="dslim/bert-large-NER")
        try:
            if query in self.query_cache:
                result = self.query_cache[query]
                elapsed_time = 0
                prompt = result
                combined_text = ''
            else:
                entities_extracted = entity_extractor.extract(text=query)
                self.main_logger.info(entities_extracted)

                # Execute the query and store the result in the cache
                start_time = time.time()
                combined_text, logging_result = self.execute_query(query, entities_extracted)
                elapsed_time = time.time() - start_time
                self.query_cache[query] = combined_text

                if cpu:
                    print("[-] No GPU Available...")
                    self.generate_answer(query, combined_text)
                else:
                    print("[+] GPUs Available ...")
                    prompt, combined_text = self.set_prompt(combined_text, query)
                    print("[+] Retrieval Prompt Process completed")

            if prompt is not None:
                print("[+] Prompt is not empty ...")
                return prompt, combined_text
            else:
                return query

        except KeyboardInterrupt:
            print("Reset by the User")
            self.main_logger.warning("RESET")
        except Exception as e:
            self.main_logger.error(f"An error occurred: {e}")

    def extract_exploit_section(self, combined_text):
        """Extract the exploit section from the combined text."""
        if "EXPLOIT:" in combined_text:
            exploit_result = combined_text.split("EXPLOIT:", 1)[1].split("Infos:")[0]
            exploit_result = exploit_result.replace("EXPLOIT:", "")
            if not exploit_result and "METASPLOIT:" in combined_text:
                exploit_result = combined_text.split("METASPLOIT:", 1)[1].split("EXPLOIT:")[0]
                exploit_result = exploit_result.replace("METASPLOIT:", "")
            return exploit_result
        else:
            return ""

    def set_emergency_prompt(self, combined_text, query, second_emergency=False):
        """Set the emergency prompt based on the combined text and query."""
        print("[!] Cached Retrieval -> Setting EMERGENCY PROMPT ...")
        if second_emergency:
            try:
                print("[:|] SECOND EMERGENCY !!!")
                exploit_section = self.extract_exploit_section(combined_text)
                if exploit_section:
                    print("[!] Exploit Exfiltration")
                    prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                    Answer the Question leveraging both Code Snippets and Contextual Information.
                    Always insert and incorporate all the Code Snippets in the response.
                    Code Snippets:

                    {exploit_section} 

                    Question: {query}[/INST]"""
                else:
                    print("[!] Info Exfiltration")
                    infos = self.extract_infos(combined_text)
                    prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                        Leverage the Contextual Information to craft informed responses.
                        Incorporate code if present in Contextual Information.
                        Contextual Information:
                        [Expert] {infos} [/Expert]
                        Question: {query}[/INST]"""
            except Exception as e:
                print("[-] Exception in set_emergency_prompt LEVEL 2 -> ", e)
                time.sleep(5)
        else:
            print("[:|] FIRST EMERGENCY !!!")
            try:
                print("[!] Trying Code Exfiltration ...")
                code_available = self.extract_code_available(combined_text)
                print(code_available)
                if code_available:                    
                    prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                    Answer the Question leveraging both Code Snippets and Contextual Information.
                    You must insert all the available Code Snippets in the response as examples related to the Question.
                    Code Snippets:

                    {code_available} 

                    Question: {query}[/INST]"""
                else:
                    print("[!] Info Exfiltration")
                    infos = self.extract_infos(combined_text)
                    print(infos)
                    prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                        Leverage the Contextual Information to craft informed responses.
                        Incorporate code if present in Contextual Information.
                        Contextual Information:
                        [Expert] {infos} [/Expert]
                        Question: {query}[/INST]"""
            except Exception as e:
                print("[-] Exception in set_emergency_prompt -> ", e)
                time.sleep(5)
        return prompt

    def code_empty_check(self, code_content):
        """Check if the code content is empty."""
        keywords = ["METASPLOIT:", "EXPLOIT:", "LOCAL:"]
        try:
            for word in keywords:
                if word in code_content:
                    result_code = code_content.replace(word, "")
            if not result_code.strip():
                return False
            if result_code.strip():
                return True
        except Exception as e:
            print("[-] ERROR in code_empty_check -> ", e)
            return False

    def extract_code_available(self, combined_text):
        """Extract the available code section from the combined text."""
        if "Code Available" in combined_text:
            try:
                result_text = combined_text.split("Code Available:", 1)[1].split("Infos:")[0]
                is_code = self.code_empty_check(result_text)
                if not is_code:
                    result_text = ""
            except Exception as e:
                print("[-] ERROR in extract_code_available >> result_text -> ", e)
                result_text = combined_text.split("Infos:")[0]
            return result_text
        else:
            return ""

    def extract_infos(self, combined_text):
        """Extract the information section from the combined text."""
        try:
            return combined_text.split("Infos:", 1)[1]
        except Exception as e:
            print("[-] ERROR IN extract_infos -> ", e)
            return 'No Infos Available'

    def generate_answer(self, query, combined_text):
        """Generate an answer based on the combined text and query."""
        try:
            prompt = self.set_prompt(combined_text, query)

            if prompt is not None:
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    generated_ids = self.model.generate(
                        **inputs,
                        streamer=self.streamer,
                        max_length=30000,
                        temperature=0.3,
                    )

                    outputs = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )

                except RuntimeError as cuda_memory_error:
                    self.handle_cuda_memory_error(
                        cuda_memory_error, combined_text, query
                    )
            else:
                self.main_logger.warning(
                    "Prompt is None. Unable to generate an answer."
                )

        except Exception as e:
            self.main_logger.error(f"An error occurred in generate_answer: {e}")

    def handle_cuda_memory_error(self, cuda_memory_error, combined_text, query):
        """Handle CUDA memory errors during answer generation."""
        self.main_logger.error(f"CUDA out of memory error: {cuda_memory_error}")
        print("\nATTENTION: CUDA out of memory. Try with a smaller input.")
        torch.cuda.empty_cache()

        emergency_prompt = self.set_emergency_prompt(combined_text, query)
        gpu_memory_after_empty_cache = torch.cuda.max_memory_allocated()
        print(
            f"GPU Memory after empty_cache: {gpu_memory_after_empty_cache / 1024 / 1024:.2f} MB"
        )

        if emergency_prompt is not None:
            inputs = self.tokenizer(emergency_prompt, return_tensors="pt").to("cuda:4")

            try:
                generated_ids = self.model.generate(
                    **inputs, streamer=self.streamer, max_length=30000, temperature=0.3
                )

                outputs = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            except RuntimeError as emergency_memory_error:
                self.main_logger.error(
                    f"EMERGENCY 2: Emergency CUDA out of memory error: {emergency_memory_error}"
                )
                print(
                    "\nEMERGENCY 2: Emergency CUDA out of memory. Unable to generate an answer."
                )
                torch.cuda.empty_cache()
                second_emergency_prompt = self.set_emergency_prompt(
                    combined_text, query, second_emergency=True
                )
                if second_emergency_prompt is not None:
                    second_inputs = self.tokenizer(
                        second_emergency_prompt, return_tensors="pt"
                    ).to("cpu")

                    try:
                        second_generated_ids = self.model.generate(
                            **second_inputs,
                            streamer=self.streamer,
                            max_length=30000,
                            temperature=0.3,
                        )

                        second_outputs = self.tokenizer.batch_decode(
                            second_generated_ids, skip_special_tokens=True
                        )
                    except RuntimeError as second_emergency_memory_error:
                        self.main_logger.error(
                            f"EMERGENCY 3: Second Emergency CUDA out of memory error: {second_emergency_memory_error}"
                        )
                        print(
                            "\nEMERGENCY 3: Second Emergency CUDA out of memory. Unable to generate an answer."
                        )

    def set_prompt(self, combined_text, query):
        """Set the prompt for generating an answer."""
        try:
            code_available = self.extract_code_available(combined_text)
            infos = self.extract_infos(combined_text)
            print("--------------------------------------------SET PROMPT-------------------------------------------------")
            print(code_available)
            print('Infos: ')
            print(infos)
            if code_available:
                print("1")
                prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                Answer the Question using both Code Snippets and Contextual Information.
                Contextual Information:

                {infos} 

                Code Snippets:

                {code_available} 
                
                Provide a detailed response to the following question:
                Question: {query}[/INST]"""
            elif infos:
                print("Operation -> 2")
                print(infos)
                prompt = f"""[INST] You are an Expert in the field of cybersecurity,
                Answer the Question using the Contextual Information.
                Contextual Information:
                {infos} 
                Provide a detailed response to the following question:
                Question: {query}[/INST]"""
                print("--------------------PROMPT--------------------------")
                print(prompt)
            else:
                print("3")
                context = self.cybersecurity_assistant.run(query)
                prompt = f"""[INST] As a distinguished expert in the field of cybersecurity,
                your extensive knowledge positions you as a highly sought-after authority capable of addressing intricate challenges. 
                Leverage the provided insights to craft informed responses, incorporating code snippets if present in the context or asked.
                Contextual Information:
                {context} 
                Question: {query}[/INST]"""
                
            return prompt, combined_text
        except Exception as e:
            print(f"Error in set_prompt: {e}")
            self.main_logger.error(f"Error in set_prompt: {e}")
            return None

    def execute_query(self, query, entities_extracted):
        """Execute the query by running multiple retrieval systems in parallel."""
        try:
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                # Execute multiple retrieval systems in parallel
                question_future = executor.submit(self.question_retrieval_system.run, query, entities_extracted)
                code_future = executor.submit(self.code_retrieval_system.generate_context, query, entities_extracted)
                entity_future = executor.submit(self.entity_retrieval_system.retrieve_entities, query)
                malware_future = executor.submit(self.malware_analyzer.run, query)

            code_result = code_future.result()
            question_result = question_future.result()
            entity_result = entity_future.result()
            malware_result = malware_future.result()
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            print("Thread Time Execution:", elapsed_time)

            combined_text = ""

            if code_result and code_result not in [
                {"METASPLOIT": None, "EXPLOIT": None},
                {"METASPLOIT": "", "EXPLOIT": ""},
            ]:
                combined_text += "Code Available:\n"

                code_dict = code_result
                if code_dict["METASPLOIT"]:
                    st.sidebar.code(code_dict["METASPLOIT"])
                if code_dict["EXPLOIT"] and code_dict["EXPLOIT"] != code_dict["METASPLOIT"]:
                    st.sidebar.divider()
                    st.sidebar.code(code_dict["EXPLOIT"])
                code_content = "\n".join(
                    f"{key}: {value}" for key, value in code_dict.items() if value
                )
                combined_text += f"{code_content}\n\n"

                combined_text += "\n"
            
            combined_text += f"Infos:\n{question_result}\n\n{entity_result}\n\n{malware_result}"

            logging_result = {
                "Code Retrieval": code_result,
                "Question Retrieval": question_result,
                "Entity Retrieval": entity_result,
                "Malware Retrieval": malware_result
            }

            return combined_text, logging_result
        except Exception as e:
            self.main_logger.error(f"Error in execute_query: {e}")
            return None, None

    def log_result(self, operation, result, elapsed_time):
        """Log the result of an operation."""
        try:
            self.main_logger.info(f"{operation} Result:\n{result}")
            self.main_logger.info(f"Time {operation}: {elapsed_time} seconds")
        except Exception as e:
            self.main_logger.error(f"Error in log_result: {e}")


if __name__ == "__main__":
    # Instantiate and run the DataRetrievalSystem
    data_retrieval_system = DataRetrievalSystem()
    while True:
        query = input("\nInsert a query: ")
        data_retrieval_system.run(query, cpu=False)
