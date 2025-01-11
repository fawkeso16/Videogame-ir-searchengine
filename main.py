from retrieval import read_html_from_folder, process_file_and_save, load_and_process_files, fix_metadata
from processing import tokenize_query, compute_cosine_similarity, tokenize_texts, calculate_document_weights, getInvertedIndex, getMetadata, calculate_query_weights, generate_normalized_weighted_matrix, normalize_document_weights
from testing import cosine_graph, precision_recall_graph


import pickle
import spacy
import numpy as np
from spacy.pipeline import EntityRuler
nlp = spacy.load("en_core_web_sm")


metadata_dict = {}

# Setting up NER
with open('Videogame-ir-searchengine/processed_game_entities.pkl', 'rb') as f:
    processed_game_entities = pickle.load(f)

ruler = nlp.add_pipe("entity_ruler", before="ner")

def create_patterns(entities_dict):
    patterns = []
    for label, values in entities_dict.items():
        for value in values:
            patterns.append({"label": label, "pattern": value})
    return patterns

patterns = create_patterns(processed_game_entities)
ruler.add_patterns(patterns)


with open('Videogame-ir-searchengine/relevant_docs.pkl', 'rb') as f:
    relevant_docs = pickle.load(f)
with open('Videogame-ir-searchengine/metadata_dict.pkl', 'rb') as f:
    meta = pickle.load(f)


def main():
    html_folder = "/Users/oliverfawkes/Downloads/videogames 2"
    output_folder = "/Users/oliverfawkes/Downloads/videogames 2/videogames_processed"
    num_files_to_process = 399

    # html_files = read_html_from_folder(html_folder)

    # print(html_files)
    # for html_file in html_files[:num_files_to_process]: 
    #     process_file_and_save(html_folder, html_file, output_folder)

#     metadata = load_and_process_files(output_folder, num_files_to_process)

#     fixed = fix_metadata(metadata)
   
    tokenize_texts(meta)
    weights = calculate_document_weights()
    normalized = normalize_document_weights(weights)

    print(getInvertedIndex())
    testqueries = {
    "testquery1": "ICO",
    "testquery2": "Okami",
    "testquery3": "Devil Kings",
    "testquery4": "Dynasty Warriors",
    "testquery5": "Sports Genre Games",
    "testquery6": "Hunting Genre Games",
    "testquery7": "Game Developed by Eurocom",
    "testquery8": "Game Published by Activision",
    "testquery9": "Game Published by Sony Computer Entertainment",
    "testquery10": "Teen game ps2",
 }

    precision_recall_graph(testqueries, relevant_docs, normalized)


main()

