from retrieval import read_html_from_folder, process_file_and_save, load_and_process_files, fix_metadata
from processing import tokenize_texts, calculate_document_weights, normalize_document_weights, getMetadata
from testing import precision_recall_graph_tests, do_query_single
import config
import os

def main():
    meta = config.metadata
    tokenize_texts(meta)
    weights = calculate_document_weights()
    normalized = normalize_document_weights(weights)

    # print(getMetadata())
    done = False
    while done != True: 
        query = str(input("Enter query: (enter exit to quit): "))
        if query.lower() == "exit":
            done = True
            break
        do_query_single(query, normalized)



#pickle file use
def Method1():
    meta = config.metadata
    relevant_docs = config.relevant_docs
    processed_game_entites = config.processed_game_entities
    tokenize_texts(meta)
    weights = calculate_document_weights()
    normalized = normalize_document_weights(weights)

    testqueries = {
    "testquery1": "ICO",
    "testquery2": "Okami",
    "testquery3": "Devil Kings",
    "testquery4": "Dynasty Warriors",
    "testquery5": "Sports Genre Games",
    "testquery6": "Hunting Genre Games",
    "testquery7": "Game Developed by Eurocom",
    "testquery8": "cars game",
    "testquery9": "game from 2000 october",
    "testquery10": "plane game",
 }
    precision_recall_graph_tests(testqueries, relevant_docs, normalized)


#all in one method, will need relevant docs pickle file.
def Method2():
    relevant_docs = config.relevant_docs

    html_folder = config.html_folder
    output_folder = config.output_folder
    num_files_to_process = 399

    if not os.path.exists(html_folder):
        raise FileNotFoundError(f"The folder '{html_folder}' does not exist. Please check the folder is in your Downloads directory.")

    os.makedirs(output_folder, exist_ok=True)
    
    html_files = read_html_from_folder(html_folder)
    for html_file in html_files[:num_files_to_process]: 
        process_file_and_save(html_folder, html_file, output_folder)

    metadata = load_and_process_files(output_folder, num_files_to_process)
    fixed = fix_metadata(metadata)
    tokenize_texts(fixed)
    weights = calculate_document_weights()
    normalized = normalize_document_weights(weights)


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
    "testquery10": "Games from 2004",
 }

    precision_recall_graph_tests(testqueries, relevant_docs, normalized)

# Method1()
# Method2()
main()


