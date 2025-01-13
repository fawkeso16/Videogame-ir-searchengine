import os
import pickle
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])  


# folder path (adjust this to your directory)
# user = 'oliverfawkes'
folder = 'Videogame-ir-searchengine'

# Load processed_game_entities from file
def load_processed_game_entities(folder):
    try:
        with open(os.path.join(folder, 'processed_game_entities.pkl'), 'rb') as f:
            processed_game_entities = pickle.load(f)
        print("processed_game_entities.pkl loaded successfully.")
        return processed_game_entities
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file is missing: {e.filename}. Ensure it is present in the '{folder}' folder.")

# Load relevant_docs from file
def load_relevant_docs(folder):
    try:
        with open(os.path.join(folder, 'relevant_docs.pkl'), 'rb') as f:
            relevant_docs = pickle.load(f)
        print("relevant_docs.pkl loaded successfully.")
        return relevant_docs
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file is missing: {e.filename}. Ensure it is present in the '{folder}' folder.")

# Load metadata_dict from file
def load_metadata_dict(folder):
    try:
        with open(os.path.join(folder, 'metadata_dict.pkl'), 'rb') as f:
            meta = pickle.load(f)
        print("metadata_dict.pkl loaded successfully.")
        return meta
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file is missing: {e.filename}. Ensure it is present in the '{folder}' folder.")
def create_patterns(entities_dict):
    patterns = []
    for label, values in entities_dict.items():
        for value in values:
            patterns.append({"label": label, "pattern": value})
    return patterns


processed_game_entities = load_processed_game_entities(folder)
metadata = load_metadata_dict(folder)
patterns = create_patterns(processed_game_entities)
relevant_docs = load_relevant_docs(folder)

home_dir = os.path.expanduser("~")
html_folder = os.path.join(home_dir, "Downloads", "videogame")
output_folder = os.path.join(html_folder, "videogames_processed")
