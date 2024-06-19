from threading import Thread, Lock, Event
from queue import Queue
from flask_socketio import SocketIO
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch
from flask import Flask, request, jsonify
import subprocess
import os 
import traceback
import logging
import signal
import sys
import gc
import requests
import time

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
thread_pid = None
exception_queue = Queue()
inputs = None
INTERRUPT_FLAG = False
text = ""

# URLs for server communication
server_url = 'http://ip_address:6000'
end_url = 'http://sender_receiver_ip:5000'

# Log file path
log_file_path = 'SERVER.log'  
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    """
    Custom stopping criteria for model generation based on INTERRUPT_FLAG.
    """
    global INTERRUPT_FLAG
    return INTERRUPT_FLAG

def configure_model():
    """
    Configure and load the model with 4-bit quantization.
    """
    print("[+] Loading Model ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model

def configure_tokenizer():
    """
    Configure and load the tokenizer.
    """
    print("[!] Loading Tokenizer ...")
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    return tok

def configure_streamer():
    """
    Configure and load the text streamer.
    """
    print("[!] Loading Streamer ...")
    streamer = TextIteratorStreamer(tok)
    return streamer

def clear_specific_gpu_memory(gpu_ids):
    """
    Clear the GPU memory for specific GPU IDs.
    """
    gc.collect()
    for gpu_id in gpu_ids:
        try:
            with torch.cuda.device(gpu_id):
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[+] GPU {gpu_id} memory cleared.")
        except Exception as e:
            print(f"[-] Unable to clear GPU {gpu_id} memory: {e}")

def cleanup(signum, frame, start):
    """
    Clean up resources and clear GPU memory before exit.
    """
    print("\n[+] Cleaning up before exit...")
    gpu_ids = [4, 5]
    
    # Force clear GPU cache multiple times
    for _ in range(3):
        clear_specific_gpu_memory(gpu_ids)
        torch.cuda.empty_cache()

    if not start:
        print("[+] Exiting...")
        show_gpu('GPU at closing, after Cleaning')
        try:
            socketio.stop()
        except RuntimeError as e:
            print("[!] Exception in cleanup -> ", e)
        sys.exit(0)

# Register the cleanup function for termination signals
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

def handle_exception(exception):
    """
    Handle exceptions by logging the error.
    """
    error = f"[!] SERVER Exception -> {exception}"
    print(error)
    logging.error(error)

def show_gpu(msg):
    """
    Display GPU usage information.
    """
    print(msg)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            gpu_info_lines = result.stdout.strip().split('\n')
            
            print("\nGPU INFO:")
            print("--------------------------------------------------------------------------------")
            print("Index |  Memory Used   |  Memory Free  |  GPU Utilization  |  Memory Utilization")
            print("--------------------------------------------------------------------------------")
            
            for line in gpu_info_lines:
                index, memory_used, memory_free, gpu_utilization, memory_utilization = map(int, line.split(','))
                if index == 4 or index == 5:
                    print(f"{index:5} | {memory_used:11} MB | {memory_free:11} MB | {gpu_utilization:15}% | {memory_utilization:18}%")
            print("\n")
        else:
            print(f"Error in executing nvidia-smi: {result.stderr}")
    
    except Exception as e:
        print(f"Error: {e}")

def get_pid():
    """
    Get and print the PID of the current thread.
    """
    global thread_pid
    thread_pid = os.getpid()
    print(f"Thread PID: {thread_pid}")

def resetting_interrupt_flag():
    """
    Reset the INTERRUPT_FLAG and reconfigure the streamer.
    """
    global INTERRUPT_FLAG
    if INTERRUPT_FLAG:
        print("[+] Resetting INTERRUPT FLAG -> False")
        logging.info("[+] Resetting INTERRUPT FLAG -> False")
        INTERRUPT_FLAG = False
        configure_streamer()
    return

def less_used_gpu():
    """
    Find the GPU with the least memory usage.
    """
    gpu_memory = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    device = torch.device(f"cuda:{gpu_memory.index(min(gpu_memory))}" if torch.cuda.is_available() else "cpu")
    return device

def llm_generation(prompt):
    """
    Generate text using the LLM with the given prompt.
    """
    global inputs, INTERRUPT_FLAG, model, tok, streamer, text
    resetting_interrupt_flag()
    stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
    try:
        inputs = tok([prompt], return_tensors="pt").to('cuda:4')
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=32000, repetition_penalty=1.1, temperature=0.7, stopping_criteria=stopping_criteria)
        thread = Thread(target=model_generate_with_exception_handler, args=(generation_kwargs,), daemon=True)
        thread.start()
        show_gpu(' ---------  GPU memory usage before streaming ---------------')
        print("[*] MAIN THREAD -> Streaming response ...")
        logging.info("[*] MAIN THREAD -> Streaming response ...")
        access = False
        if not INTERRUPT_FLAG:
            for new_text in streamer:
                if access:
                    params = {'text': new_text}
                    print(new_text)
                    response = requests.post(f'{end_url}/output', json=params)
                if '[/INST]' in new_text:
                    access = True
        access = False
        resetting_interrupt_flag()
        thread.join()  # Wait for the thread to finish
        print("[+] MAIN THREAD -> Streaming Completed. OK")
        logging.info("[+] MAIN THREAD -> Streaming Completed. OK")
        del inputs
        torch.cuda.synchronize()  # Synchronize the GPU
        torch.cuda.empty_cache()
        print("[+] GPU Memory Occupied by Prompt Released.")
        show_gpu('--------- GPU memory usage after clearing cache ---------------')

    except Exception as e:
        print("[!] ERROR in llm_generation", e)
        logging.error("[:(] MAIN THREAD -> An error occurred: %s", e)
        handle_exception(e)
        exception_queue.put(e)

def model_generate_with_exception_handler(generation_kwargs):
    """
    Generate model output with exception handling.
    """
    global INTERRUPT_FLAG, model, inputs
    try:
        model.generate(**generation_kwargs)
    except Exception as e:
        print("[!] ERROR in model_generate_with_exception_handler", e)
        if "CUDA out of memory" in str(e):
            error = "!!! CHILD THREAD -> [:(] CUDA out of memory\n"
            print(error)
            logging.error(error)
            requests.post(f'{end_url}/output', json={'text': error})
            torch.cuda.synchronize()  # Synchronize the GPU
            torch.cuda.empty_cache()
        if "KeyboardInterrupt" in str(e):
            print("[!!!] STOPPING GENERATION")
            logging.error("[!] STOP GENERATION -> model_generate_with_exception_handler")
            handle_exception(error)
            exception_queue.put(error)
        else:
            logging.error("CHILD THREAD -> [:(] An error occurred: %s", e)
            handle_exception(str(e))
            exception_queue.put(str(e))
        logging.warning("CHILD THREAD -> [!] Exception handled, continuing with the thread.")

def send_prompt_and_follow_file(prompt):
    """
    Send the prompt to the LLM and follow the output file for updates.
    """
    if prompt.strip():
        print("[-] Sending prompt to LLM ...")
        logging.info("[-] Sending prompt to LLM ...")
        llm_generation(prompt)

@app.route('/generate_output', methods=['POST'])
def generate_output():
    """
    Endpoint to receive a prompt and generate output.
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        print("[+] Received Prompt : ", prompt)
        logging.info(f"[+] Received Prompt : {prompt[:500]}")
        try:
            question = prompt.split("Question:", 1)[1].split("[/INST]")[0]
            print(question)
            logging.info(f"[+] Received Question -> {question}")
        except Exception as e:
            print("[-] Error in Finding Question ...")
        send_prompt_and_follow_file(prompt)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(e)
        logging.error("[!!!] ERROR in generate_output -> ", e)
        return jsonify({'status': 'failed'}), 404

@app.route('/is_reachable', methods=['POST'])
def is_server_reachable():
    """
    Endpoint to check if the server is reachable.
    """
    try:
        logging.info("SERVER IS REACHABLE -> OK")
        return jsonify({'status': 'SERVER: reachable'}), 200
    except Exception as e:
        print("[!] Error in is_server_reachable", e)
        logging.error("[!] Error in is_server_reachable")
        return jsonify({'status': 'SERVER: unreachable'}), 404

@app.route('/interrupt', methods=['POST'])
def interrupt():
    """
    Endpoint to handle interrupt requests.
    """
    global INTERRUPT_FLAG
    print("[+] Starting Interruption ...")
    logging.warning("[+] Starting Interruption ...")
    try:
        if not INTERRUPT_FLAG:
            INTERRUPT_FLAG = True
            print("[+] Setting INTERRUPT FLAG -> ", INTERRUPT_FLAG)
            logging.info("[+] Setting INTERRUPT FLAG -> True")
            return jsonify({'status': '[->] SERVER: No Child Process Running.'}), 200
        else:
            print("[+] Interrupt already DONE")
            logging.warning("[+] Interrupt already DONE")
            return jsonify({"[+] Interrupt already DONE"}), 200
    except Exception as e:
        print("ERROR in interrupt: ", e)
        return jsonify({'status': '[->] SERVER: ERROR in Interruption.'}), 404

print(f"Main Process PID: {os.getpid()}")
model = configure_model()
tok = configure_tokenizer()
streamer = configure_streamer()
show_gpu('Initial GPU memory usage:')

if __name__ == "__main__":
    try:
        print("[+] Server is Running ...")
        socketio.run(app, host='0.0.0.0', port=6000)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
