# Author : Oliver Fawkes
# Description : file to process text using spacy, store in an inverted index, create a tf-idf weighted(normlaized) matrix and calculate cosine similarity between 2 vectors
# History : 
#15/12/2024 - v1.0 - methods to tokenise a text input and create an inverted index
#16/12/2024 - v1.1 - processing inverted index, creating useful methods such as get document count etc
#18/12/2024 - v1.2 - calcuate tf-idf weights and any methods needed this caluclation
#19/12/2024 - v1.3 - calculate normalized weights of querys and documents, added method to compute cosine similarity- bugs  
#20/12/2024 - v1.3.1 - bug fixes 

import spacy
import math
import pandas as pd

nlp = spacy.load("en_core_web_sm")
inverted_index_dict = {}

weights = {}
unique_terms = set()  

def addToInvertedIndex(text, doc):
    for token in text:
        if token not in inverted_index_dict:
            inverted_index_dict[token] = {"doc_frequency": {}, "total_tf": 0}

        if doc not in inverted_index_dict[token]["doc_frequency"]:
            inverted_index_dict[token]["doc_frequency"][doc] = 0

        inverted_index_dict[token]["doc_frequency"][doc] += 1
        inverted_index_dict[token]["total_tf"] += 1


#useful methods for main functionality
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
#useful methods end 



#tokenising text methods
def tokenise_texts(texts, i):
    # efficiency chnage - pipe texts instead of calling each iteration
    docs = nlp.pipe(texts)
    for doc in docs:
        filtered_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
        addToInvertedIndex(filtered_tokens, i)

def tokenise_query(query):
    doc = nlp(query)
    filtered_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop
    ]

    return filtered_tokens


#calculating tf-idf weight and normalizing methods#
def calculate_document_weights(count):
    weights = {}  
    doc_count = count  

    for token, term_data in inverted_index_dict.items():
        doc_frequency = term_data["doc_frequency"]
        
        df = len(doc_frequency)
        if df == 0:
            continue  
        idf = math.log(doc_count / df)  
    
        for doc, tf in doc_frequency.items():
            term_frequency = tf / term_data["total_tf"]  
            tf_idf = term_frequency * idf
            if doc not in weights:
                weights[doc] = {}
            weights[doc][token] = tf_idf
    return weights

def normalize_weights(weights):
    normalized_weights = {}
    
    for doc_id in weights:
        magnitude = math.sqrt(sum(weight ** 2 for weight in weights[doc_id].values()))
        normalized_weights[doc_id] = {
            term: (weight / magnitude) if magnitude != 0 else 0
            for term, weight in weights[doc_id].items()
        }
    return normalized_weights


def calculate_query_weights(count, query):
    query_tokens = tokenise_query(query)  
    query_weights = {}
    for term in query_tokens:
        tf = math.log(1+ query_tokens.count(term)) 
        idf = get_doc_count_term_frequency(term)  
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


#matrix generation
def generate_normalized_weighted_matrix(weights):
    df = pd.DataFrame.from_dict(weights, orient="index")
    df = df.transpose()
    df = df.fillna(0)
    return df


#dot product of 2 vectors, 
def compute_cosine_similarity(query_weights, doc_weights):
    cosine_similarities = {}
    query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    if query_magnitude == 0:
        return None, 0.0

    for doc_id, term_weights in doc_weights.items():
        doc_magnitude = math.sqrt(sum(weight ** 2 for weight in term_weights.values()))
        if doc_magnitude == 0:
            cosine_similarities[doc_id] = 0.0 
            continue
        
        dot_product = sum(query_weights.get(term, 0) * term_weights.get(term, 0) for term in query_weights
        )

        if query_magnitude != 0 and doc_magnitude != 0:
            cosine_similarities[doc_id] = dot_product / (query_magnitude * doc_magnitude)
        else:
            cosine_similarities[doc_id] = 0.0 

    best_doc_id = max(cosine_similarities, key=cosine_similarities.get, default=None)
    best_similarity = cosine_similarities.get(best_doc_id, 0.0)
    
    return best_doc_id, best_similarity


#getters for testing 
def get_unique_terms():
    return unique_terms

def getInvertedIndex():
    return inverted_index_dict
