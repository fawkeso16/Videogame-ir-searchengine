# Date : 12 december 2024 
# Author : Oliver Fawkes
# Description : contains methods to caluclate weights of documetns using tf-idf weighting via an inverted index, and compare them to a weighted query and return ranked reusults.
# History : 
#  12/12/2024 - v1 .00 - intilal methods added, inverted index creation, and method to populate
#  16/12/2024 - v1 .10  - tokenizing and lemmatizing text mthods added, helper methods added for future calulations
#  17/12/2024 - v1. 20 -  document and query weghting and normalization added
#  18/12/2024 - v1. 30 -  cosine similarity added
#  24/12/2024 - v1. 35 -  Bug fixes and structure change - added metadata structure, altered current methods, added improved boost to weights for metadata
#  27/12/2024 - v1. 40 - many more 'logic' bug fixes, changed boosts 

import re
import spacy
import math
import pandas as pd

nlp = spacy.load("en_core_web_sm")
inverted_index_dict = {}
metadata_dict = {}  
weights = {}
unique_terms = set()  


#weigth boosts
TITLE_BOOST = 30
GENRE_BOOST = 50
PUBLISHER_BOOST = 50
DEVELOPER_BOOST = 50
RATING_BOOST = 50
RELEASE_DATE_BOOST = 10


def addToInvertedIndex(text, docname, metadata):
    if docname not in metadata_dict:
        metadata_dict[docname] = metadata

    for token in text:
        if token not in inverted_index_dict:
            inverted_index_dict[token] = {"doc_frequency": {}, "total_tf": 0}

        if docname not in inverted_index_dict[token]["doc_frequency"]:
            inverted_index_dict[token]["doc_frequency"][docname] = {"tf": 0}

        inverted_index_dict[token]["doc_frequency"][docname]["tf"] += 1
        inverted_index_dict[token]["total_tf"] += 1  


# Useful methods for main functionality
def get_term_frequency(term):
    return inverted_index_dict.get(term, {}).get("total_tf", 0)

def get_total_document_term_count(doc):
    total = 0
    for term_data in inverted_index_dict.values():
        total += term_data["doc_frequency"].get(doc, {}).get("tf", 0)
    return total

def get_total_documents():
    totalDocs = {docID for term_data in inverted_index_dict.values() for docID in term_data["doc_frequency"]}
    return len(totalDocs)

def get_doc_count_term_frequency(term):
    if term not in inverted_index_dict:
        return 0
    return len(inverted_index_dict[term]["doc_frequency"])

def get_all_terms_in_document(document_id):
    terms_in_document = []
    for term, term_data in inverted_index_dict.items():
        if document_id in term_data["doc_frequency"]:
            terms_in_document.append(term)
    return terms_in_document


#methods too tokenise documents and tokenise querys, using spacy. returns an array of filtered tokens
def tokenize_texts(allFiles, metadata_dict):
    texts = list(allFiles.values())  
    file_names = list(allFiles.keys())  
    texts.extend(file_names)
    texts = [' '.join(text) if isinstance(text, list) else text for text in texts] 
    
    docs = nlp.pipe(texts)
    for doc, file_name in zip(docs, file_names):

        filtered_tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        
        split_file_name = file_name.lower().split('-') 
        filtered_tokens.extend(split_file_name) 
        filtered_tokens.append(file_name)    
    
        addToInvertedIndex(filtered_tokens, file_name, metadata_dict[file_name])


def tokenize_query(query):
    doc = nlp(query)
    filtered_tokens = [token.lemma_.lower() for token in doc 
                       if not token.is_punct and not token.is_space 
                       and not token.is_stop]
    
    print("filtered query : ", filtered_tokens)
    
    return filtered_tokens


# Calculating tf-idf weight and normalizing methods, used slighly altered tf formula when working with querys
def calculate_document_weights():
    weights = {}  
    doc_count = get_total_documents()

    for token, term_data in inverted_index_dict.items():
        doc_frequency = term_data["doc_frequency"]
        
        df = len(doc_frequency)
        if df == 0:
            continue  
        idf = math.log(1 + (doc_count / (1 + df)))
    
        for doc, tf in doc_frequency.items():
            term_frequency = math.log(1 + tf["tf"]) / term_data["total_tf"]
            tf_idf = term_frequency * idf

            #boost metadata checks 
            if token in metadata_dict[doc]["title"]:  
                tf_idf *= TITLE_BOOST  
            elif token in metadata_dict[doc]["genre"]:
                tf_idf *= GENRE_BOOST
            elif token in metadata_dict[doc]["developer"]:
                tf_idf *= DEVELOPER_BOOST
            elif token in metadata_dict[doc]["publisher"]:
                tf_idf *= PUBLISHER_BOOST
            elif token in metadata_dict[doc]["rating"]:
                tf_idf *= RATING_BOOST
            elif token in metadata_dict[doc]["releaseDate"]:
                tf_idf *= RELEASE_DATE_BOOST

            if doc not in weights:
                weights[doc] = {}
            weights[doc][token] = tf_idf
    return weights


def normalize_document_weights(weights):
    normalized_weights = {}
    
    for doc_id in weights:
        magnitude = math.sqrt(sum(weight ** 2 for weight in weights[doc_id].values()))
        normalized_weights[doc_id] = {
            term: (weight / magnitude) if magnitude != 0 else 0
            for term, weight in weights[doc_id].items()
        }
    return normalized_weights



def calculate_query_weights(query, entity_matches, matching_keywords):
    print(f"Calculating weights for: {query}")

    count = get_total_documents()
    query_weights = {}
    for term in query:
        tf = math.log(1 + query.count(term)) 
        idf = get_doc_count_term_frequency(term)  
        inverse_document_frequency = math.log(count / idf) if idf > 0 else 0

        weight = tf * inverse_document_frequency
        print(f"\nTerm: {term}")
        print(f"Initial weight: {weight}")

        #boost metadata checks 
        if any(term in metadata["title"] for metadata in metadata_dict.values()):
            weight *= TITLE_BOOST
            print(f"Title boost applied: {weight}")
        elif any(term in metadata["genre"] for metadata in metadata_dict.values()):
            weight *= GENRE_BOOST
            print(f"Genre boost applied: {weight}")
        elif any(term in metadata["developer"] for metadata in metadata_dict.values()):
            weight *= DEVELOPER_BOOST
            print(f"Developer boost applied: {weight}")
        elif any(term in metadata["publisher"] for metadata in metadata_dict.values()):
            weight *= PUBLISHER_BOOST
            print(f"Publisher boost applied: {weight}")
        elif any(term in metadata["rating"] for metadata in metadata_dict.values()):
            weight *= RATING_BOOST
            print(f"Rating boost applied: {weight}")
        elif any(term in metadata["releaseDate"] for metadata in metadata_dict.values()):
            weight *= RELEASE_DATE_BOOST
            print(f"Release date boost applied: {weight}")
        else:
            print("No boost applied")

        query_weights[term] = weight
    return query_weights


def normalize_query_weights(query_weights):
    magnitude = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    
    normalized_query = {
        term: (weight / magnitude) if magnitude != 0 else 0
        for term, weight in query_weights.items()
    }
    return normalized_query

# Matrix generation
def generate_normalized_weighted_matrix(weights):
    df = pd.DataFrame.from_dict(weights, orient="index")
    df = df.transpose()
    df = df.fillna(0)
    return df


#methods to calculate dot product of 2 (preferably normlazied) vectors.
def compute_cosine_similarity(query_weights, doc_weights):
    cosine_similarities = {}
    query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    if query_magnitude == 0:
        return [], 0.0  # Return an empty list and 0 relevance if the query has no terms

    # Compute cosine similarities
    for doc_id, term_weights in doc_weights.items():
        doc_magnitude = math.sqrt(sum(weight ** 2 for weight in term_weights.values()))
        if doc_magnitude == 0:
            cosine_similarities[doc_id] = 0.0
            continue
        
        dot_product = sum(query_weights.get(term, 0) * term_weights.get(term, 0) for term in query_weights)
        cosine_similarities[doc_id] = dot_product / (query_magnitude * doc_magnitude)

    # Filter docs with a score > 0.05
    filtered_docs = [(doc_id, score) for doc_id, score in cosine_similarities.items() if score > 0.01]

    # Sort filtered docs by cosine similarity
    sorted_filtered_docs = sorted(filtered_docs, key=lambda item: item[1], reverse=True)

    # Limit to top 10 results
    top_docs = sorted_filtered_docs[:10]

    return top_docs


def precision_ten(doc_array, relevant_docs_for_query):
    if not doc_array:
        return 0
    # Count how many relevant documents are in the top N retrieved documents
    relevant_count = sum(1 for doc_id, _ in doc_array if doc_id in relevant_docs_for_query)
    return relevant_count / len(doc_array)


def calculate_precision_recall(retrieved_docs, relevant_docs_for_query, top_n=10):
    # Limit to the top N retrieved documents
    retrieved_docs_top_n = retrieved_docs[:top_n]
    
    # Count how many relevant documents are in the top N retrieved documents
    relevant_count = sum(1 for doc_id, _ in retrieved_docs_top_n if doc_id in relevant_docs_for_query)

    # Calculate precision: relevant retrieved docs / total retrieved docs (top_n)
    precision = relevant_count / top_n if len(retrieved_docs_top_n) > 0 else 0
    
    # Calculate recall: relevant retrieved docs / total relevant docs (capped at 10)
    recall = relevant_count / min(len(relevant_docs_for_query), 10) if len(relevant_docs_for_query) > 0 else 0

    return precision, recall



# Getters for testing 
def get_unique_terms():
    return unique_terms

def getInvertedIndex():
    return inverted_index_dict

def getMetadata():
    return metadata_dict


#mostly used for debugging 
def lemmatize_metadata(metadata_dict):
    processed_dict = {}
    
    for filename, metadata in metadata_dict.items():
        processed_dict[filename] = {}
        for field, value in metadata.items():
            value = ' '.join(str(value).split())
            doc = nlp(value)
            processed_value = ' '.join([token.lemma_.lower() for token in doc if token is not token.is_punct])
            processed_dict[filename][field] = processed_value
    
    return processed_dict