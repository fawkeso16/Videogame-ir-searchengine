from retrieval import read_html_from_csv, process_file_and_save, load_files
from processing import compute_cosine_similarity, tokenise_texts, calculate_document_weights, getInvertedIndex, getMetadata, calculate_query_weights, generate_normalized_weighted_matrix, normalize_weights
from query import search_query 


import os
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")

allFiles = {}
metadata_dict = {}

directory = "/Users/oliverfawkes/Downloads/videogames"
# file_paths = read_html_from_csv('Videogame-ir-searchengine/videogame-labels.csv')
# for path in file_paths:  # Loop through each file path
#     process_file_and_save(path, directory)  #
 

pickle_data = load_files(directory, 399)

file_amount = 399
# Loop through the specified number of files
for i, (filename, data) in enumerate(pickle_data.items()):
    if i >= file_amount:
        print(f"too many {i}")
        break  
    game_info = data  
    base_filename = filename.replace('.html.pkl', '')

    # Use the base_filename as the key
    allFiles[base_filename] = game_info["description"]
    metadata_dict[base_filename] = game_info["metadata"]
    


tokenise_texts(allFiles, metadata_dict)



# print(getMetadata())
#pickle file, 1-title, 2-developer, 3-publisher, 4-genre, 5-rating, 6-description


weights = calculate_document_weights(10)
normalize = normalize_weights(weights)

matrix = generate_normalized_weighted_matrix(normalize)

# print(matrix)
# #query = str(input("Enter query: "))

terms_dict = getInvertedIndex()
search_terms = ["ico", "okami", "devil", "kings"]

for key in terms_dict:
    if key.lower() in [term.lower() for term in search_terms]:
        print(f"Found key: {key}")

doc = nlp("ICO okami")
for token in doc:
    print(f"Text: {token.text}, Lemma: {token.lemma_}, Is stop: {token.is_stop}")


testquery1 = "ICO"
testquery2 = "Okami"
testquery3 = "Devil Kings"
testquery4 = "Dynasty Warriors"
testquery5 = "Sports Genre Games"
testquery6 = "Hunting Genre Games"
testquery7 = "Game Developed by Eurocom"
testquery8 = "Game Published by Activision"
testquery9 = "Game Published by Sony Computer Entertainment"
testquery10 = " Teen PS2 Games"

# weighted_query = search_query(testquery1)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery2)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery3)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery4)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery5)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery6)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery7)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery8)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery9)
# print(compute_cosine_similarity(weighted_query, weights))
# weighted_query = search_query(testquery10)
# print(compute_cosine_similarity(weighted_query, weights))


