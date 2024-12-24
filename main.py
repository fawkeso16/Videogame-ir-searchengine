from retrieval import read_html_from_csv, process_file_and_save, load_files
from processing import compute_cosine_similarity, tokenise_texts, calculate_document_weights, getInvertedIndex, getMetadata, calculate_query_weights, generate_normalized_weighted_matrix, normalize_weights
from query import search_query 


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
    


tokenise_texts(allFiles, metadata_dict)

# print(allFiles)

# print(getMetadata())
#pickle file, 1-title, 2-developer, 3-publisher, 4-genre, 5-rating, 6-description


weights = calculate_document_weights(399)
normalize = normalize_weights(weights)

matrix = generate_normalized_weighted_matrix(normalize)


print(getMetadata())
# print(matrix)
# #query = str(input("Enter query: "))

# terms_dict = getInvertedIndex()
# search_terms = ["ico", "okami", "devil", "kings", "warriors"]

# doc = nlp(" ".join(search_terms))

# # Use SpaCy attributes to filter
# filteredterms = [
#     token.lemma_.lower()
#     for token in doc
#     if not token.is_punct and not token.is_space and not token.is_stop
# ]


# row_labels = matrix.index.tolist()  # This gets the row labels (terms)

# # Check if any filtered term occurs in the row labels
# rows_with_terms = [row for row in row_labels if any(filtered_term in row.lower() for filtered_term in filteredterms)]

# # Print the rows containing any of the filtered terms
# print(f"Rows containing any of the filtered terms: {rows_with_terms}")

# print(filteredterms)

# print(columns)

# search_string = ""

# # Check rows
# is_in_rows = matrix.isin([search_string]).any(axis=1).any()

# # Check columns
# is_in_columns = matrix.isin([search_string]).any(axis=0).any()

# print(f"Found in rows: {is_in_rows}, Found in columns: {is_in_columns}")



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

# print(getMetadata())
weighted_query = search_query(testquery1)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery2)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery3)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery4)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery5)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery6)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery7)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery8)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery9)
print(compute_cosine_similarity(weighted_query, weights))
weighted_query = search_query(testquery10)
print(compute_cosine_similarity(weighted_query, weights))


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

data3 = {
    "Testquery1": ("ico", 0.8939709316260098),
    "Testquery2": ("okami", 0.8299052743697891),
    "Testquery3": ("devil-kings", 0.23060358612300125),
    "Testquery4": ("dynasty-warriors-4", 0.3643998478574397),
    "Testquery5": ("tony-hawks-underground", 0.14077938868936918),
    "Testquery6": ("tom-clancys-rainbow-six-3", 0.027650539948905987),
    "Testquery7": ("spyro-v-working-title", 0.3786790752598778),
    "Testquery8": ("tony-hawks-underground", 0.06761183402473288),
    "Testquery9": ("sega-ages-classics-collection", 0.04688995322160524),
    "Testquery10": ("manhunt-2", 0.010114075057013743),
 }
# # Extract queries and similarity scores
queries = list(data3.keys())
scores = [item[1] for item in data3.values()]

print(np.mean(scores))

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
