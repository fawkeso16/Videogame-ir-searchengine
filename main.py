from retrieval import read_html_from_csv, process_file_and_save, load_files
from processing import tokenize_query, calculate_precision_recall, lemmatize_metadata, compute_cosine_similarity, tokenize_texts, calculate_document_weights, getInvertedIndex, getMetadata, calculate_query_weights,precision_ten, generate_normalized_weighted_matrix, normalize_document_weights
from query import search_query 


import csv
import os
import pickle
import matplotlib.pyplot as plt
import spacy
import numpy as np
from spacy.pipeline import EntityRuler
import re
nlp = spacy.load("en_core_web_sm")

allFiles = {}
metadata_dict = {}


#Setting up NER
with open('processed_game_entities.pkl', 'rb') as f:
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


with open('relevant_docs.pkl', 'rb') as f:
    relevant_docs = pickle.load(f)
with open('metadata_dict.pkl', 'rb') as f:
    meta = pickle.load(f)


directory = "/Users/oliverfawkes/Downloads/videogames"
# file_paths = read_html_from_csv('videogame-labels.csv')
# for path in file_paths:  # Loop through each file path
#     process_file_and_save(path, directory)  #
 

#read all files and get data
# pickle_data = load_files(directory, 399)
# file_amount = 399
# for i, (filename, data) in enumerate(pickle_data.items()):
#     if i >= file_amount:
#         print(f"too many {i}")
#         break  
#     game_info = data  
#     base_filename = filename.replace('.html.pkl', '')
#     title_words = game_info["title"] or ""
#     metadata_words = " ".join(game_info["metadata_no_struc"] or [])
#     description_words = game_info["description"] or ""
#     all_words = f"{title_words} {metadata_words}".split()
#     allFiles[base_filename] = all_words
#     metadata_dict[base_filename] = game_info["metadata"]
    

# lemmatized_metadata = lemmatize_metadata(metadata_dict) 
# def clean_metadata(metadata_dict):
#     for file_name, game_metadata in metadata_dict.items():
#         for field, field_value in game_metadata.items():
#             if isinstance(field_value, str):
#                 cleaned_value = re.sub(r"\b's\b", '', field_value)
#                 cleaned_value = re.sub(r'[^\w\s]', '', cleaned_value)
#                 cleaned_value = re.sub(r'\s*ps2\s*', '', cleaned_value, flags=re.IGNORECASE)
#                 cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
#                 game_metadata[field] = cleaned_value

#     return metadata_dict

# def clean_developer_and_publisher(data_dict):
#     regex = r'\s*\(.*?\)\s*'  
#     for key in ['DEVELOPER', 'PUBLISHER']:
#         if key in data_dict:
#             cleaned_list = []
#             for item in data_dict[key]:
#                 cleaned_item = re.sub(regex, '', item).strip()
#                 cleaned_list.append(cleaned_item)
#             data_dict[key] = cleaned_list
#     return data_dict

# print(meta)
tokenize_texts(meta)
weights = calculate_document_weights()
normalize = normalize_document_weights(weights)


# def simplify_ratings(rating):
#     rating_lower = rating.lower()  # Convert to lowercase for uniformity
#     if "teen" in rating_lower:
#         return "teen"
#     elif "everyone" in rating_lower:
#         return "everyone"
#     elif "mature" in rating_lower:
#         return "mature"
#     elif "pending" in rating_lower:
#         return "rating pending"
#     elif "tba" in rating_lower:
#         return "tba"
#     else:
#         return rating_lower  # Return the original rating if no match


# def clean_titles(titles):
#     cleaned_titles = []
#     for title in titles:
#         # Convert to lowercase
#         title = title.lower()
        
#         # Remove punctuation
#         title = re.sub(r'[^\w\s]', '', title)
        
#         # Replace double spaces with a single space
#         title = re.sub(r'\s+', ' ', title)
        
#         # Combine terms like 'hawk s' to 'hawks'
#         title = re.sub(r'\s+s\b', 's', title)
        
#         cleaned_titles.append(title)
#     return cleaned_titles

# ratings2 = simplify_ratings(ratings)
# processed_game_entities["RATING"] = ratings2

# for game, details in meta.items():
#     simplified_rating = simplify_ratings(details['rating'])  
#     details['rating'] = simplified_rating  # Update the game's rating field

# ratings = [rating.lower() for rating in processed_game_entities["RATING"]]    

# print(ratings)
# # titles = clean_titles(processed_game_entities["TITLE"])

# processed_game_entities["RATING"] = ratings



# print(processed_game_entities)

# print(meta)
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

# # # Initialize the plot
# plt.figure(figsize=(10, 7))

# print(processed_game_entities)

for query_key, query_value in testqueries.items():
    print(f"Processing Query: {query_key} - {query_value}") 
    relevant_docs_for_query = relevant_docs[query_key]    
    weighted_query = search_query(query_value)
    print(weighted_query)
    retrieved_docs = compute_cosine_similarity(weighted_query, weights)
    
    # print(f"Relevant Docs for {query_key}: {relevant_docs_for_query}")  
    print(f"Retrieved Docs for {query_key}: {retrieved_docs[:10]}...")  

    precision = precision_ten(retrieved_docs, relevant_docs_for_query)
    recall = calculate_precision_recall(retrieved_docs, relevant_docs_for_query)[1]  
    
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}") 
    # label_text = f"{query_key}: Precision={precision:.2f}, Recall={recall:.2f}"  
    # plt.scatter(recall, precision, marker="o", label=label_text)
# print(processed_game_entities["PUBLISHER"])
# print(processed_game_entities['RATING'])

# print(processed_game_entities)

# sets = {}

# # Loop through the dictionary and handle duplicates by splitting strings and creating sets
# for key, values in processed_game_entities.items():
#     # Split values by comma, strip whitespace, and convert to a set
#     flattened_values = {val.strip() for item in values for val in item.split(',')}
#     sets[key] = flattened_values

# # Print the resulting sets
# print(sets)

# with open('processed_game_entities.pkl', 'wb') as f:
#     pickle.dump(sets, f)


# metadata = getMetadata()

# Print metadata for all documents in the list
# debug_files = [
#     'call-of-duty-finest-hour',
#     'wakeboarding-unleashed-featuring-shaun-murray',
#     'x-men-legends',
#     'x-men-z-axis',
#     'shrek-superslam',
#     'spider-man-trilogy',
#     'true-crime-streets-of-la',
#     'gun',
#     'tony-hawks-underground',
#     'tony-hawks-underground-2'
# ]

# for file_name, game_metadata in metadata.items():
#     if file_name in debug_files:
#         print(f"\nMetadata for {file_name}:")
#         for field, value in game_metadata.items():
#             print(f"{field}: {value}")
# # # Customize the plot
# plt.title("Precision-Recall Points for All Queries")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.grid(True)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')

# plt.tight_layout()  
# plt.show()


# data5 = {
#     "Testquery1": ('ico', 0.994638054774928),
#     "Testquery2": ('okami', 0.9571932634657468),
#     "Testquery3": ('devil-kings', 0.43613882140538535),
#     "Testquery4": ('dynasty-warriors-4', 0.8089893791643922),
#     "Testquery5": ('mlb-06-the-show', 0.19840834811182861),
#     "Testquery6": ('cabelas-big-game-hunter', 0.984193009227017),
#     "Testquery7": ('james-bond-007-nightfire', 0.9733904578069024),
#     "Testquery8": ('x-men-z-axis', 0.5754608701782999),
#     "Testquery9": ('rise-to-honor', 0.2785146736489353),
#     "Testquery10": ('the-simpsons', 0.031452981678974465),
#  }

# # # # # Extract queries and similarity scores
# queries = list(data5.keys())
# scores = [item[1] for item in data5.values()]

# print(np.mean(scores))

# # # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(queries, scores, marker='o', linestyle='-', color='b', label="Cosine Similarity")
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
# plt.xlabel("Test Queries")
# plt.ylabel("Cosine Similarity Score")
# plt.title("Cosine Similarity Scores for Test Queries")
# plt.ylim(0, 1)  # Adjust as needed for better visualization
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()

# # Show the plot
# plt.show()
