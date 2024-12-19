from retrieval import read_html_from_csv, process_file_and_save, load_files
from processing import tokenise_texts, getInvertedIndex, get_total_document_term_count, get_total_documents, get_doc_count_term_frequency, get_term_frequency, calculate_weighting, get_unique_terms

import pandas as pd

# Get list of file paths from your CSV
# file_paths = read_html_from_csv('videogame-labels.csv')

# # Process all HTML files.
# for path in file_paths:  # Loop through each file path
#     process_file_and_save(path)  #

directory = "/Users/oliverfawkes/Downloads/videogames"  
allFiles = load_files(directory, 100)




for i, data in enumerate(allFiles):
    #print(f"Data from pickle file {i+1}, {data}")  
    tokenise_texts(data, i)

#inv_index = getInvertedIndex()

weights = calculate_weighting(100)

df = pd.DataFrame.from_dict(weights, orient='index')

# Fill NaN values with 0 (for terms that don't exist in certain documents)
df = df.fillna(0)

print(df)




#print(getInvertedIndex())
# print(get_total_document_term_count(3))
# print(get_total_documents())
# print(get_doc_count_term_frequency("run"))
# Print or process the extracted text

