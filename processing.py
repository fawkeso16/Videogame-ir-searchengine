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
#  29/12/2024 - v1. 50 - Implemetnted NER structure, big changes to all methods, tokenising text - added entity matching fully altered layout, removed redundant processing of text as chnaged to use mtadata soley
#  - alterted weighting for both query and document to account for exact matches of enittys

import re
import spacy
import math
import pandas as pd
import pickle
from nltk import PorterStemmer

stemmer = PorterStemmer()
keywords = ["title", "genre", "developer", "publisher", "release date", "rating"]
stemmed_keywords = [stemmer.stem(word) for word in keywords]
unwanted_words = ["game", "games", "ps2"]
nlp = spacy.load("en_core_web_sm", disable=["ner"])  



inverted_index_dict = {}
metadata_dict = {}  
weights = {}
unique_terms = set()  


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


#weigth boosts
TITLE_BOOST = 30
GENRE_BOOST = 30
PUBLISHER_BOOST = 30
DEVELOPER_BOOST = 30
RATING_BOOST = 30
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

def clean_text(text):
    return ''.join([char.lower() for char in text if char.isalnum() or char.isspace()])

def tokenize_texts(metadata_dict):
    for file_name, game_metadata in metadata_dict.items():
        filtered_tokens, tokens = [], []

        for field, value in game_metadata.items():
            if isinstance(value, str) and value.strip():
                tokens.extend(value.strip().lower().split())

        doc = nlp(' '.join(tokens))
        entity_matches = {ent.text.lower(): ent.label_ for ent in doc.ents}

        for entity_key in entity_matches.keys():
            if entity_key not in tokens:
                tokens.append(entity_key)
            entity_terms = entity_key.split()
            for term in entity_terms:
                if term not in tokens:
                    tokens.append(term)

        for token in tokens:
            if token in entity_matches:
                filtered_tokens.append(token)
                entity_terms = token.split()
                if len(entity_terms) > 1:
                    filtered_tokens.extend(entity_terms)
            else:
                filtered_tokens.append(token)

        metadata_dict[file_name]['tokens'] = filtered_tokens
        addToInvertedIndex(filtered_tokens, file_name, metadata_dict[file_name])



def tokenize_query(query):
    doc = nlp(query)
    filtered_tokens = [
        stemmer.stem(token.lemma_.lower()) if token.lemma_.lower().endswith('s') else token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space and not token.is_stop
        and token.lemma_.lower() not in unwanted_words
        and token.lemma_.lower() not in keywords
        and stemmer.stem(token.lemma_.lower()) not in stemmed_keywords
    ]
    
    # print("filtered query : ", filtered_tokens)
    return filtered_tokens


# Calculating tf-idf weight and normalizing methods, used slighly altered tf formula when working with querys
def calculate_document_weights():
    entity_match = ''
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

            if token == doc:  
                tf_idf *= 2
                
            token_lower = token.lower()
            
            if token_lower == metadata_dict[doc]["title"].lower():
                tf_idf *= TITLE_BOOST
                entity_match = token_lower
            elif token_lower == metadata_dict[doc]["genre"].lower():
                tf_idf *= GENRE_BOOST
                entity_match = token_lower
            elif token_lower == metadata_dict[doc]["developer"].lower():
                tf_idf *= DEVELOPER_BOOST
                entity_match = token_lower
            elif token_lower == metadata_dict[doc]["publisher"].lower():
                tf_idf *= PUBLISHER_BOOST
                entity_match = token_lower
            elif token_lower == metadata_dict[doc]["rating"].lower():
                tf_idf *= RATING_BOOST
                entity_match = token_lower
            elif token_lower == metadata_dict[doc]["releaseDate"].lower():
                tf_idf *= RELEASE_DATE_BOOST
                entity_match = token_lower

            if doc not in weights:
                weights[doc] = {}
            weights[doc][token] = tf_idf

    for doc, terms in weights.items():
        additional_weights = {}
        for token, weight in terms.items():
            sub_terms = token.split()  
            for term in sub_terms:
                term_weight = weight * 10
                if term not in additional_weights:
                    additional_weights[term] = term_weight
                else:
                    additional_weights[term] += term_weight

        for term, weight in additional_weights.items():
            if term not in weights[doc]:
                weights[doc][term] = weight
            else:
                weights[doc][term] += weight

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



def calculate_query_weights(query, synonyms):
    print(f"Calculating weights for: {query}")

    count = get_total_documents()
    query_weights = {}

    for term in query:
        tf = math.log(1 + query.count(term))  # Term Frequency
        idf = get_doc_count_term_frequency(term)  
        inverse_document_frequency = math.log(count / idf) if idf > 0 else 0

        weight = tf * inverse_document_frequency
        print(f"\nTerm: {term}")
        print(f"Initial weight: {weight}")

        if term in synonyms:
            weight *= 0.2
            print(f"Synonym penalty applied to {term}: {weight}")

        for entity_type, values in processed_game_entities.items():
            if any(term.lower() == value.lower() for value in values):
                weight *= 2  
                print(f"Exact entity match boost applied to {term}: {weight}")
                break  

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

    filtered_docs = [(doc_id, score) for doc_id, score in cosine_similarities.items() if score > 0.1]

    sorted_filtered_docs = sorted(filtered_docs, key=lambda item: item[1], reverse=True)

    top_docs = sorted_filtered_docs[:10]

    return top_docs




def precision_ten(doc_array, relevant_docs_for_query):
    # Validate input data types
    if not isinstance(doc_array, list) or not isinstance(relevant_docs_for_query, list):
        return 0  # Return 0 if the input is invalid

    if not doc_array:
        return 0
    
    relevant_count = sum(1 for doc_id, _ in doc_array if isinstance(doc_id, str) and doc_id in relevant_docs_for_query)
    
    return relevant_count / len(doc_array) if len(doc_array) > 0 else 0


def calculate_precision_recall(retrieved_docs, relevant_docs_for_query, top_n=10):
    # Validate input data types
    if not isinstance(retrieved_docs, list) or not isinstance(relevant_docs_for_query, list):
        return 0, 0  # Return (0, 0) if the input is invalid

    retrieved_docs_top_n = retrieved_docs[:top_n]
    
    # Validate retrieved_docs_top_n contains tuples of (doc_id, score)
    relevant_count = sum(1 for doc_id, _ in retrieved_docs_top_n if isinstance(doc_id, str) and doc_id in relevant_docs_for_query)
    
    precision = relevant_count / top_n if len(retrieved_docs_top_n) > 0 else 0
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