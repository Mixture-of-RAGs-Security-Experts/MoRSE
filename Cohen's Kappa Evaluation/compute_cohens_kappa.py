import re
import json
from sklearn.metrics import cohen_kappa_score

# Function to read the file and create a dictionary for the first format
def create_dict_from_file1(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = r'\d+\.\s\[(.*?)\]\s(.*?)(?:,|\?|\.|$)'
    matches = re.findall(pattern, content)

    dictionary = {}
    for match in matches:
        judgment, question = match
        dictionary[question.strip()] = judgment.strip()

    return dictionary

# Function to read the file and create a dictionary for the second format
def create_dict_from_file2(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    dictionary = {}
    pattern = r'\[(.*?)\]\s(.*)'
    for line in content:
        match = re.match(pattern, line)
        if match:
            judgment, question = match.groups()
            dictionary[question.strip()] = judgment.strip()

    return dictionary

# Function to map textual judgments to numerical categories
def map_categories(dictionary):
    category_map = {
        "Correct": 0,
        "Partially Correct": 1,
        "Incorrect": 2,
        "Irrelevant": 3
    }
    return [category_map.get(judgment, -1) for judgment in dictionary.values()]

# Specify the paths of the text files
file_path1 = 'Review1.txt'
file_path2 = 'Review2.txt'

# Create the dictionaries
dict_questions1 = create_dict_from_file2(file_path1)
dict_questions2 = create_dict_from_file2(file_path2)

# Find the common keys
common_keys = set(dict_questions1.keys()).intersection(set(dict_questions2.keys()))

# Extract the values corresponding to the common keys
categories1 = [dict_questions1[key] for key in common_keys]
categories2 = [dict_questions2[key] for key in common_keys]

# Map the textual judgments to numerical categories
categories1_mapped = map_categories(dict(zip(common_keys, categories1)))
categories2_mapped = map_categories(dict(zip(common_keys, categories2)))

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(categories1_mapped, categories2_mapped)
print(f"Cohen's Kappa: {kappa_score}")

# Print the common keys and categories for verification
print("Common keys:")
print(common_keys)
print("Categories 1:")
print(categories1_mapped)
print("Categories 2:")
print(categories2_mapped)

# Save the dictionaries to JSON files
with open('questions_judgments1.json', 'w', encoding='utf-8') as json_file1:
    json.dump(dict_questions1, json_file1, ensure_ascii=False, indent=4)

with open('questions_judgments2.json', 'w', encoding='utf-8') as json_file2:
    json.dump(dict_questions2, json_file2, ensure_ascii=False, indent=4)
