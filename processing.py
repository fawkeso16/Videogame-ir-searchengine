# Date : 15 December 2024
# Author : Oliver Fawkes
# Description : methods to  tokenise a text input and create an inverted index(v1.0 - v1.1), calcuate tf-idf weights(v1.2) and any methods needed this caluclation

# History : 15/12/2024 - v1.0, 16/12/2024 - v1.1, 18/12/2024 - v1.2,

import spacy
import math
import pandas as pd
import numpy as np


nlp = spacy.load("en_core_web_sm")


inverted_index_dict = {}

#dict to hold weights for each term in each document.
weights = {}
unique_terms = set()  # Set to store unique terms across all documents


def addToInvertedIndex(text, docnum):
    for token in text:
        if token not in inverted_index_dict:
            inverted_index_dict[token] = {"doc_frequency": {}, "total_tf": 0}
        
        if docnum not in inverted_index_dict[token]["doc_frequency"]:
            inverted_index_dict[token]["doc_frequency"][docnum] = 0
        
        inverted_index_dict[token]["doc_frequency"][docnum] += 1
        inverted_index_dict[token]["total_tf"] += 1


# Total frequency of the term across all documents
def get_term_frequency(term):
    return inverted_index_dict.get(term, {}).get("total_tf", 0)


def get_total_document_term_count(docID):
    total = 0
    for term_data in inverted_index_dict.values():
        total += term_data["doc_frequency"].get(docID, 0)
    return total


def get_total_documents():
    totalDocs = {docID for term_data in inverted_index_dict.values() for docID in term_data["doc_frequency"]}
    return len(totalDocs)


# Frequency of terms in the document
def get_doc_count_term_frequency(term):
    if term not in inverted_index_dict:
        return 0
    return len(inverted_index_dict[term]["doc_frequency"])


def tokenise_texts(texts, i):
    #pipe texts instead of calling each iteration
    docs = nlp.pipe(texts)
    for doc in docs:
        filtered_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
        addToInvertedIndex(filtered_tokens, i)


def get_all_terms_in_document(document_id):
    
    terms_in_document = []

    # Iterate through the inverted index to check each term's frequency in the document
    for term, term_data in inverted_index_dict.items():
        # Check if the document_id is in the doc_frequency for the current term
        if document_id in term_data["doc_frequency"]:
            terms_in_document.append(term)

    return terms_in_document



def calculate_weighting(docs):
    for document in range(docs):
        #print(f"Processing document {document}")
        #print(get_all_terms_in_document(document))
        for term in inverted_index_dict:
            # Get term frequency for the current document
            tf = inverted_index_dict[term]["doc_frequency"].get(document, 0)
            df = get_total_document_term_count(document)

          #  print(f"Document {document}, Term: {term}, tf: {tf}, df: {df}")
            if df == 0: 
                continue

            # Normalize term frequency
            t_frequency = tf / df
            term_frequency = math.log(1 + t_frequency)
        

            # Get document count for the term
            idf = get_doc_count_term_frequency(term)
            #ßprint(f"Term: {term}, idf: {idf}")
            if idf == 0 or idf == docs:  
                continue

            # Ensure that IDF is calculated correctly and cannot be negative
            if idf > docs:
                idf = docs

            # Calculate inverse document frequency
            inverse_document_frequency = math.log(docs / idf)
            if inverse_document_frequency == 0: 
                continue

            # Calculate TF-IDF weight
            weight = term_frequency * inverse_document_frequency

            if document not in weights:
                weights[document] = {}  # Initialize the dictionary for the document
            weights[document][term] = weight  # Store the weight for the term

            unique_terms.add(term)

    return weights



def get_unique_terms():
    return unique_terms


def getInvertedIndex():
    return inverted_index_dict
