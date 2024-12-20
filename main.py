from retrieval import read_html_from_csv, process_file_and_save, load_files
from processing import compute_cosine_similarity, tokenise_texts, getInvertedIndex, get_total_document_term_count, get_total_documents, get_doc_count_term_frequency, get_term_frequency, calculate_document_weights, calculate_query_weights, get_unique_terms, generate_normalized_weighted_matrix, normalize_weights
from query import search_query 


import os
import pickle
# file_paths = read_html_from_csv('videogame-labels.csv')
# for path in file_paths:  # Loop through each file path
#     process_file_and_save(path)  #

file_amount = 50


directory = "/Users/oliverfawkes/Downloads/videogames"
allFiles = load_files(directory, file_amount)  # Returns a dictionary

for file_name, data in allFiles.items():
    tokenise_texts(data, file_name)



weights = calculate_document_weights(50)
normalize = normalize_weights(weights)

matrix = generate_normalized_weighted_matrix(normalize)

query = str(input("Enter query: "))

weighted_query = search_query(query)

print(compute_cosine_similarity(weighted_query, weights))

