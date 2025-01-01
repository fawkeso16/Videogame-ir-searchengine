from retrieval import read_html_from_csv, process_file_and_save, load_files
from processing import calculate_precision_recall, lemmatize_metadata, compute_cosine_similarity, tokenize_texts, calculate_document_weights, getInvertedIndex, getMetadata, calculate_query_weights,precision_ten, generate_normalized_weighted_matrix, normalize_document_weights
from query import search_query 


import csv
import os
import pickle
import matplotlib.pyplot as plt
import spacy
import numpy as np
nlp = spacy.load("en_core_web_sm")

allFiles = {}
metadata_dict = {}


directory = "/Users/oliverfawkes/Downloads/videogames"
# file_paths = read_html_from_csv('videogame-labels.csv')
# for path in file_paths:  # Loop through each file path
#     process_file_and_save(path, directory)  #
 

pickle_data = load_files(directory, 399)

file_amount = 399
for i, (filename, data) in enumerate(pickle_data.items()):
    if i >= file_amount:
        print(f"too many {i}")
        break  

    game_info = data  
    base_filename = filename.replace('.html.pkl', '')

    # Use the base_filename as the key
    title_words = game_info["title"] or ""
    metadata_words = " ".join(game_info["metadata_no_struc"] or [])
    description_words = game_info["description"] or ""

    all_words = f"{title_words} {metadata_words} {description_words}".split()
    allFiles[base_filename] = all_words
    metadata_dict[base_filename] = game_info["metadata"]
    



# game_entities = {
#     "TITLE": sorted(all_titles),
#     "GENRE": sorted(all_genres),
#     "DEVELOPER": sorted(all_developers),
#     "PUBLISHER": sorted(all_publishers),
#     "YEAR": [str(year) for year in range(1980, 2025)] 
# }


with open('game_entities.pkl', 'rb') as f:
    game_entities = pickle.load(f)


print(game_entities)
lemmatized_metadata = lemmatize_metadata(metadata_dict) 

tokenize_texts(allFiles,lemmatized_metadata)
weights = calculate_document_weights()
normalize = normalize_document_weights(weights)
matrix = generate_normalized_weighted_matrix(normalize)


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
    "testquery10": "teen games ps2"
 }


with open('relevant_docs.pkl', 'rb') as f:
    relevant_docs = pickle.load(f)



# Initialize the plot
plt.figure(figsize=(10, 7))


# Example loop to process queries and generate precision-recall plot
# for query_key, query_value in testqueries.items():
#     print(f"Processing Query: {query_key} - {query_value}")  # Debugging output for the current query

#     # Get the relevant documents for the current query
#     relevant_docs_for_query = relevant_docs[query_key]  # Access the relevant docs list using the query key    

#     # Assume you have the `search_query` and `compute_cosine_similarity` functions
#     weighted_query = search_query(query_value, game_entities, all_titles)
    
#     # Compute cosine similarities and get top retrieved docs
#     retrieved_docs = compute_cosine_similarity(weighted_query, weights)
    
#     # Print out some of the retrieved docs for debugging
#     print(f"Relevant Docs for {query_key}: {relevant_docs_for_query}")  # Debugging output for relevant docs
#     print(f"Retrieved Docs for {query_key}: {retrieved_docs[:10]}...")  # Only show the first 10 for brevity

#     # Calculate precision and recall for the current query
#     precision = precision_ten(retrieved_docs, relevant_docs_for_query)
#     recall = calculate_precision_recall(retrieved_docs, relevant_docs_for_query)[1]  # Only need recall value
    
#     print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")  # Debugging output for precision and recall
    
#     # Plot precision-recall point for the current query
#     label_text = f"{query_key}: Precision={precision:.2f}, Recall={recall:.2f}"  # Create a label with precision and recall
#     plt.scatter(recall, precision, marker="o", label=label_text)

# # Customize the plot
# plt.title("Precision-Recall Points for All Queries")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.grid(True)
# plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')

# plt.tight_layout()  
# plt.show()



# data = {
#     "Testquery1": ("ICO", (None, 0.0)),
#     "Testquery2": ("Okami", (None, 0.0)),
#     "Testquery3": ("Devil Kings", ('king-of-fighters-xi.html.pkl', 0.09606044701998305)),
#     "Testquery4": ("Dynasty Warriors", ('rogue-trooper.html.pkl', 0.20488449621233762)),
#     "Testquery5": ("Sports Genre Games", ('major-league-baseball-2k6.html.pkl', 0.04148868785887241)),
#     "Testquery6": ("Hunting Genre Games", ('crash-nitro-kart.html.pkl', 0.22890176981963825)),
#     "Testquery7": ("Game Developed by Eurocom", ('james-bond-007-nightfire.html.pkl', 0.17011874478275632)),
#     "Testquery8": ("Game Published by Activision", ('wakeboarding-unleashed-featuring-shaun-murray.html.pkl', 0.10545440070219074)),
#     "Testquery9": ("Game Published by Sony Computer Entertainment", ('ratchet-and-clank-up-your-arsenal.html.pkl', 0.03159301016204154)),
#     "Testquery10": ("Teen PS2 Games", ('ratchet-and-clank-up-your-arsenal.html.pkl', 0.009052509412378423)),
# }

# # Example data
# data2 = {
#     "Testquery1": ("ico", 0.3344732577028222),
#     "Testquery2": ("okami", 0.3024563068324231),
#     "Testquery3": ("devil-may-cry-2", 0.10525557099149994),
#     "Testquery4": ("dynasty-warriors-4", 0.23406403887318303),
#     "Testquery5": ("mx-2002-featuring-ricky-carmichael", 0.014836496255056682),
#     "Testquery6": ("dead-to-rights", 0.024672187418736113),
#     "Testquery7": ("spyro-v-working-title", 0.07665168928851937),
#     "Testquery8": ("sega-ages-classics-collection", 0.0637760436079248),
#     "Testquery9": ("sega-ages-classics-collection", 0.06083201765376797),
#     "Testquery10": ("manhunt-2", 0.0021135675471584477),
# }

# data3 = {
#     "Testquery1": ("ico", 0.8939709316260098),
#     "Testquery2": ("okami", 0.8299052743697891),
#     "Testquery3": ("devil-kings", 0.23060358612300125),
#     "Testquery4": ("dynasty-warriors-4", 0.3643998478574397),
#     "Testquery5": ("tony-hawks-underground", 0.14077938868936918),
#     "Testquery6": ("tom-clancys-rainbow-six-3", 0.027650539948905987),
#     "Testquery7": ("spyro-v-working-title", 0.3786790752598778),
#     "Testquery8": ("tony-hawks-underground", 0.06761183402473288),
#     "Testquery9": ("sega-ages-classics-collection", 0.04688995322160524),
#     "Testquery10": ("manhunt-2", 0.010114075057013743),
#  }

# data4 = {
#     "Testquery1": ('ico', 0.8434373018758231),
#     "Testquery2": ('okami', 0.6266734911984305),
#     "Testquery3": ('devil-kings', 0.6036812944250204),
#     "Testquery4": ('dynasty-warriors-4', 0.26672898640300274),
#     "Testquery5": ('ea-sports-fight-night-2004', 0.09358479426705091),
#     "Testquery6": ('cabelas-big-game-hunter', 0.035286635625949504),
#     "Testquery7": ('spyro-v-working-title', 0.6980927728105496),
#     "Testquery8": ('tony-hawks-underground', 0.1860707913232095),
#     "Testquery9": ('wild-arms-3', 0.1142751012807096),
#     "Testquery10": ("devil-kings", 0.0170727852991542),
#  }

# # # # Extract queries and similarity scores
# queries = list(data4.keys())
# scores = [item[1] for item in data4.values()]

# print(np.mean(scores))

# # Plotting
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
