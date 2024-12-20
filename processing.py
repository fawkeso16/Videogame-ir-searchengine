# Date : 15 December 2024
# Author : Oliver Fawkes
# Description : methods to  tokenise a text input and create an inverted index(v1.0 - v1.1), calcuate tf-idf weights(v1.2) and any methods needed this caluclation

# History : 15/12/2024 - v1.0 || 16/12/2024 - v1.1 || 18/12/2024 - v1.2 || 19/12/2024 - v1.25

import spacy
import math
import pandas as pd
import numpy as np


nlp = spacy.load("en_core_web_sm")


inverted_index_dict = {}

#dict to hold weights for each term in each document.
weights = {}
unique_terms = set()  # Set to store unique terms across all documents


def addToInvertedIndex(text, doc):
    for token in text:
        if token not in inverted_index_dict:
            inverted_index_dict[token] = {"doc_frequency": {}, "total_tf": 0}

        if doc not in inverted_index_dict[token]["doc_frequency"]:
            inverted_index_dict[token]["doc_frequency"][doc] = 0

        inverted_index_dict[token]["doc_frequency"][doc] += 1
        inverted_index_dict[token]["total_tf"] += 1


# Total frequency of the term across all documents
def get_term_frequency(term):
    return inverted_index_dict.get(term, {}).get("total_tf", 0)


def get_total_document_term_count(doc):
    total = 0
    for term_data in inverted_index_dict.values():
        total += term_data["doc_frequency"].get(doc, 0)
    return total


def get_total_documents():
    totalDocs = {docID for term_data in inverted_index_dict.values() for docID in term_data["doc_frequency"]}
    return len(totalDocs)


# Frequency of terms in the document
def get_doc_count_term_frequency(term):
    if term not in inverted_index_dict:
        return 0
    return len(inverted_index_dict[term]["doc_frequency"])

def get_all_terms_in_document(document_id):
    terms_in_document = []

    # Iterate through the inverted index to check each term's frequency in the document
    for term, term_data in inverted_index_dict.items():
        # Check if the document_id is in the doc_frequency for the current term
        if document_id in term_data["doc_frequency"]:
            terms_in_document.append(term)

    return terms_in_document



def tokenise_texts(texts, i):
    #pipe texts instead of calling each iteration
    docs = nlp.pipe(texts)
    for doc in docs:
        filtered_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
        addToInvertedIndex(filtered_tokens, i)

def tokenise_query(query):

    doc = nlp(query)
    filtered_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop
    ]

    return filtered_tokens

def calculate_document_weights(count):
    weights = {}  # Dictionary to store the weights for all documents
    doc_count = count  # Use the passed count value instead of recalculating
    print(f"Total number of documents: {doc_count}")  # Debug print

    # Iterate over each token (term) in the inverted index
    for token, term_data in inverted_index_dict.items():
        print(f"Processing token: {token}")  # Debug print

        # Get document frequency for the current token
        doc_frequency = term_data["doc_frequency"]
        
        # Calculate inverse document frequency (IDF) for the token
        df = len(doc_frequency)
        if df == 0:
            continue  # Skip tokens that don't appear in any documents

        idf = math.log(doc_count / df)  # IDF calculation
        
        # Iterate over each document where the token appears
        for doc, tf in doc_frequency.items():
            # Calculate term frequency (TF) for the current document
            term_frequency = tf / term_data["total_tf"]  # TF is normalized by total term frequency

            # Calculate the weight for the current document and token (TF-IDF)
            tf_idf = term_frequency * idf

            # Store the weight for the document and token
            if doc not in weights:
                weights[doc] = {}
            weights[doc][token] = tf_idf

    print(f"Calculated document weights: {weights}")  # Debug print
    return weights

def normalize_weights(weights):
    # Create a dictionary to store normalized weights
    normalized_weights = {}
    
    # Calculate the magnitude for each document
    for doc_id in weights:
        magnitude = math.sqrt(sum(weight ** 2 for weight in weights[doc_id].values()))
        
        # Normalize each term's weight by dividing by the magnitude
        normalized_weights[doc_id] = {
            term: (weight / magnitude) if magnitude != 0 else 0
            for term, weight in weights[doc_id].items()
        }
    
    return normalized_weights


def calculate_query_weights(count, query):
    query_tokens = tokenise_query(query)  

    query_weights = {}
    for term in query_tokens:
        tf = math.log(1+ query_tokens.count(term))  # Term frequency in the query
        idf = get_doc_count_term_frequency(term)  # Use the inverted index to get IDF
        inverse_document_frequency = math.log(count / idf) if idf > 0 else 0

        query_weights[term] = tf * inverse_document_frequency

    return query_weights



def normalize_query_weights(query_weights):
    magnitude = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    
    normalized_query = {
        term: (weight / magnitude) if magnitude != 0 else 0
        for term, weight in query_weights.items()
    }
    
    return normalized_query




def generate_normalized_weighted_matrix(weights):

    #generate matrix with our weights and dictionary
    df = pd.DataFrame.from_dict(weights, orient="index")

    #swapping axis of matrix
    df = df.transpose()

    # Fill NaN values with 0 (for terms that don't exist in certain documents)
    df = df.fillna(0)

    return df


def compute_cosine_similarity(query_weights, doc_weights):
    cosine_similarities = {}

    # Calculate the magnitude of the query vector
    query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    if query_magnitude == 0:
        print("Query magnitude is zero, cannot compute cosine similarity.")
        return None, 0.0

    # Loop through each document to compute similarity
    for doc_id, term_weights in doc_weights.items():
        # Calculate the magnitude of the document vector
        doc_magnitude = math.sqrt(sum(weight ** 2 for weight in term_weights.values()))
        if doc_magnitude == 0:
            print(f"Document {doc_id} magnitude is zero, skipping cosine similarity calculation.")
            cosine_similarities[doc_id] = 0.0  # No similarity for this document
            continue
        
        # Compute the dot product of the query and document vectors
        dot_product = sum(
            query_weights.get(term, 0) * term_weights.get(term, 0)
            for term in query_weights
        )

        # Compute cosine similarity (avoid division by zero)
        if query_magnitude != 0 and doc_magnitude != 0:
            cosine_similarities[doc_id] = dot_product / (query_magnitude * doc_magnitude)
        else:
            cosine_similarities[doc_id] = 0.0  # Similarity is 0 if one of the vectors is 0

    # Debug print of similarities
    print(f"Cosine similarities: {cosine_similarities}")

    # Find the document with the highest similarity
    best_doc_id = max(cosine_similarities, key=cosine_similarities.get, default=None)
    best_similarity = cosine_similarities.get(best_doc_id, 0.0)
    
    return best_doc_id, best_similarity

def get_unique_terms():
    return unique_terms


def getInvertedIndex():
    return inverted_index_dict
