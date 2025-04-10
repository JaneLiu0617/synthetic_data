import random
import json
import os
import logging
from datetime import datetime
import requests

# orginal idea https://colemurray.medium.com/generating-synthetic-data-with-llms-for-fine-tuning-7d93bf271794
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OLLAMA_API_URL = "http://localhost:****/api/chat"  # Update if your Ollama server is different

def generate_synthetic_html(page_type):
    prompt = f"""
    Generate an HTML page for a {page_type} website. Make it realistic and diverse.
    Include appropriate tags, headings, and content based on the page type.
    For an informational page, include a main question and its answer.
    For a navigational page, include a navigation menu and some basic content.
    For a commercial page, include product information and a call-to-action.
    For a transactional page, include a form or a clear action for the user to take.
    Return the result as a JSON object with two keys: 'html' for the HTML content, and 'summary' for a brief description of the page.
    """
    logging.info(f"Generating synthetic HTML for {page_type} page")
    
    payload = {
        "model": "llama3.2-vision",  # Replace with your Ollama model if different
        "messages": [
            {"role": "system", "content": "You are a web developer creating diverse and realistic HTML pages."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json" # Force json output.
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90000)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        content = response.json().get("message", {}).get("content", "")
        logging.info(content)
        return json.loads(content)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from Ollama response: {e}")
        return None

# def generate_dataset(num_samples):
#     page_types = ["informational", "navigational", "commercial", "transactional"]
#     dataset = []
#     for i in range(num_samples):
#         page_type = random.choice(page_types)
#         logging.info(f"Generating sample {i+1}/{num_samples}")
#         result = generate_synthetic_html(page_type)
#         if result:
#             dataset.append((result['html'], page_type, result['summary']))
#         else:
#             logging.warning(f"Failed to generate sample {i+1}/{num_samples}")
#     return dataset
def generate_dataset(num_samples):
    page_types = ["informational", "navigational", "commercial", "transactional"]
    dataset = []
    for i in range(num_samples):
        page_type = random.choice(page_types)
        logging.info(f"Generating sample {i+1}/{num_samples}")
        result = generate_synthetic_html(page_type)
        if result:
            html_content = result.get('html')
            summary_content = result.get('summary')
            if html_content is not None and summary_content is not None:
                dataset.append((html_content, page_type, summary_content))
            else:
                logging.warning(f"Generated data for sample {i+1} is missing 'html' or 'summary': {result}")
        else:
            logging.warning(f"Failed to generate sample {i+1}/{num_samples}")
    return dataset

def write_dataset_to_file(dataset, filename):
    logging.info(f"Writing dataset to file: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logging.info(f"Dataset successfully written to {filename}")

def main():
    num_samples = 10  # Replace with your desired count
    logging.info(f"Starting synthetic data generation for {num_samples} samples")
    
    dataset = generate_dataset(num_samples)
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"synthetic_dataset_{timestamp}.json"
    
    write_dataset_to_file(dataset, filename)
    
    logging.info("Synthetic data generation complete")

if __name__ == "__main__":
    main()