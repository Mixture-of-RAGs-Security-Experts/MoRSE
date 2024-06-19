from flask import Flask, request, jsonify
import requests
import json
from threading import Thread

app = Flask(__name__)
text = ''

@app.route('/receive_query', methods=['POST'])
def receive_query():
    """
    Endpoint to receive a query and start a new thread to process it.
    """
    data = request.get_json()
    query = data.get('query')
    thread = Thread(target=send_query, args=(query,), daemon=True)
    thread.start()
    return jsonify({'query': query})

def send_query(prompt):
    """
    Function to send the query to another endpoint.
    """
    with app.app_context():
        print("[+] Sending Prompt ...")
        # Define the endpoint URL to send the query to
        url = 'http://generator_ip_address/generate_output'
        
        # Prepare the data payload with the prompt
        data_payload = {'prompt': prompt}

        try:
            # Send a POST request to the endpoint
            response = requests.post(url, json=data_payload)
            # Return the response from the other endpoint
            return jsonify({'response_from_other_endpoint': response.json()})
        except requests.exceptions.RequestException as e:
            print(f"Error sending prompt: {e}")
            return jsonify({'error': str(e)})

@app.route('/output', methods=['POST'])
def process_input_text():
    """
    Endpoint to process input text and save it to a JSON file.
    """
    global text 
    input_params = request.json  # Get parameters from the request

    # Extract the text from the request
    new_text = input_params.get('text', '')
    print(new_text, end=' ', flush=True)

    data = {
        "text": new_text
    }

    # Specify the file path for the JSON file
    file_path = 'received_text.json'

    try:
        # Open the file in write mode and save the data
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f'Data has been written to the JSON file: {file_path}')
        response_data = {'status': 'success', 'processed_text': new_text}
    except IOError as e:
        print(f"Error writing to file: {e}")
        response_data = {'status': 'error', 'error': str(e)}

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
