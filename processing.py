# Date : 12 december 2024 
# Author : Oliver Fawkes
# Description : contains methods to caluclate weights of documetns using tf-idf weighting via an inverted index, and compare them to a weighted query and return ranked reusults.
# History : 
#  12/12/2024 - v1 .00 - intitial methods added, inverted index creation, and method to populate
#  16/12/2024 - v1 .10  - tokenizing and lemmatizing text mthods added, helper methods added for future calulations
#  17/12/2024 - v1. 20 -  document and query weghting and normalization added
#  18/12/2024 - v1. 30 -  cosine similarity added
#  24/12/2024 - v1. 35 -  Bug fixes and structure change - added metadata structure, altered current methods, added improved boost to weights for metadata
#  27/12/2024 - v1. 40 - many more 'logic' bug fixes, changed boosts 
#  29/12/2024 - v1. 50 - Implemetnted NER structure, big changes to all methods, tokenising text - added entity matching fully altered layout, removed redundant processing of text as chnaged to use mtadata soley
#  - alterted weighting for both query and document to account for exact matches of enittys
#  03/01/2024 - 06/01/2024  - massive bug fixes, fixed logic and sytax issues after new implementations. mostly to do with entity matching during tokeising and weighting.

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


#Setting up NER with only custom pattern creation. lables = TITLE, GENRE, PUBLISHER, DEVLEOPER, RELEASEDATE, RATING
with open('Videogame-ir-searchengine/processed_game_entities.pkl', 'rb') as f:
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


#weight boosts, tried to scale in proportion to commonaltlity between terms
TITLE_BOOST = 5
GENRE_BOOST = 15
PUBLISHER_BOOST = 15
DEVELOPER_BOOST = 15
RATING_BOOST = 50
RELEASE_DATE_BOOST = 5


#method to populate inverted index and metadata.
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


######### Helper Methods ######### 
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
######### end of helper methods ######### 

#methodto tokenise text, to nlp text and match any entitys.
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
        filename = file_name.split('-')

        for token in filename:
            filtered_tokens.append(token)

        # print(filename)

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
    weights = {}
    doc_count = get_total_documents()

    for term, term_data in inverted_index_dict.items():
        doc_frequency = term_data["doc_frequency"]
        df = len(doc_frequency)
        if df == 0:
            continue
        idf = math.log(1 + (doc_count / (1 + df)))

        for doc_id, tf in doc_frequency.items():
            term_frequency = math.log(1 + tf["tf"]) / term_data["total_tf"]
            weight = term_frequency * idf

            entity_matches = {ent.text.lower(): ent.text for ent in nlp(term).ents if ent.label_ in processed_game_entities}
            if entity_matches:
                weight *= 3

            if any(term in metadata["title"] for metadata in metadata_dict.values()):
                weight *= TITLE_BOOST
            elif any(term in metadata["genre"] for metadata in metadata_dict.values()):
                weight *= GENRE_BOOST
            elif any(term in metadata["developer"] for metadata in metadata_dict.values()):
                weight *= DEVELOPER_BOOST
            elif any(term in metadata["publisher"] for metadata in metadata_dict.values()):
                weight *= PUBLISHER_BOOST
            elif any(term in metadata["rating"] for metadata in metadata_dict.values()):
                weight *= RATING_BOOST
            elif any(term in metadata["releaseDate"] for metadata in metadata_dict.values()):
                weight *= RELEASE_DATE_BOOST  
            if doc_id not in weights:
                weights[doc_id] = {}
            weights[doc_id][term] = weight
    return weights


def normalize_document_weights(weights):
    normalized_weights = {}
    for doc_id, doc_weights in weights.items():
        magnitude = math.sqrt(sum(weight ** 2 for weight in doc_weights.values()))
        
        if magnitude == 0:
            normalized_weights[doc_id] = {term: 0 for term in doc_weights}
        else:
            normalized_weights[doc_id] = {
                term: weight / magnitude for term, weight in doc_weights.items()
            }

    return normalized_weights


#weighting query terms, various boosts based off of entity matches and synonyms.
def calculate_query_weights(query, entity_matches):
    print(f"Calculating weights for: {query}")

    count = get_total_documents()
    query_weights = {}

    for term in query:
        tf = math.log(1 + query.count(term))  
        doc_frequency = get_doc_count_term_frequency(term)
        idf = math.log(count / (1 + doc_frequency)) if doc_frequency > 0 else 0

        weight = tf * idf

        # if term in synonyms:
        #     weight *= 0.1
            # print(f"Synonym penalty applied to {term}: {weight}")
        if term in entity_matches.values():
            weight *= 3
            # print(f"Big entity match boost applied to {term}: {weight}")
        if any(term in metadata["title"] for metadata in metadata_dict.values()):
            weight *= TITLE_BOOST
        elif any(term in metadata["genre"] for metadata in metadata_dict.values()):
            weight *= GENRE_BOOST
        elif any(term in metadata["developer"] for metadata in metadata_dict.values()):
            weight *= DEVELOPER_BOOST
        elif any(term in metadata["publisher"] for metadata in metadata_dict.values()):
            weight *= PUBLISHER_BOOST
        elif any(term in metadata["rating"] for metadata in metadata_dict.values()):
            weight *= RATING_BOOST
        elif any(term in metadata["releaseDate"] for metadata in metadata_dict.values()):
            weight *= RELEASE_DATE_BOOST

        query_weights[term] = weight
    return query_weights


def normalize_query_weights(query_weights):
    magnitude = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    if magnitude == 0:
        return {term: 0 for term in query_weights}

    normalized_weights = {
        term: weight/magnitude for term, weight in query_weights.items()
    }
    
    return normalized_weights


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
        return [], 0.0  

    for doc_id, term_weights in doc_weights.items():
        doc_magnitude = math.sqrt(sum(weight ** 2 for weight in term_weights.values()))
        if doc_magnitude == 0:
            cosine_similarities[doc_id] = 0.0
            continue
        
        dot_product = sum(query_weights.get(term, 0) * term_weights.get(term, 0) for term in query_weights)
        cosine_similarity = dot_product / (query_magnitude * doc_magnitude)
        cosine_similarities[doc_id] = cosine_similarity
        
    filtered_docs = [(doc_id, score) for doc_id, score in cosine_similarities.items() if score > 0]
    sorted_filtered_docs = sorted(filtered_docs, key=lambda item: item[1], reverse=True)
    top_docs = sorted_filtered_docs[:10]
    return top_docs


def precision_and_recall(doc_array, relevant_docs_for_query):
    if not doc_array or not relevant_docs_for_query:
        return 0, 0 
    
    docs_array = doc_array[0]
    relevant_count = 0

    relevant_docs_clean = [r.strip().lower().strip('-') for r in relevant_docs_for_query]
    
    top_docs = doc_array[:10]
    
    for doc, score in top_docs:
        doc_clean = doc.strip().lower().strip('-')
        if doc_clean in relevant_docs_clean:
            relevant_count += 1

    precision = relevant_count / min(10, len(top_docs))
    recall = relevant_count / min(10, len(relevant_docs_for_query)) 

    return precision, recall


def final_results(doc_array):
    finalranks = []
    rank = 1
    top_docs = doc_array[:10]
    for doc_name, score in top_docs:
        finalranks.append((rank, doc_name + '.html', score))  # Tuple with three elements
        rank += 1
    return finalranks



# Getters for testing 
def get_unique_terms():
    return unique_terms

def getInvertedIndex():
    return inverted_index_dict

def getMetadata():
    return metadata_dict


    
#add entites to file
def add_to_entities(key, value):
    with open('Videogame-ir-searchengine/processed_game_entities.pkl', 'rb') as f:
        game_entities = pickle.load(f)
    
    if key not in game_entities:
        raise KeyError(f"Invalid key: {key}. Valid keys are: {', '.join(game_entities.keys())}")
    
    doc = nlp(value.lower())  
    lemmatized_value = " ".join([token.lemma_ for token in doc])  
    
    game_entities[key].append(lemmatized_value)

    with open('Videogame-ir-searchengine/processed_game_entities.pkl', 'wb') as f:
        pickle.dump(game_entities, f)


def save_new_metadata_dict(dict):
    if dict:
        with open('Videogame-ir-searchengine/metadata_dict.pkl', 'wb') as f:
            pickle.dump(dict, f)


def save_new_relevant_docs(dict):
    if dict:
        with open('Videogame-ir-searchengine/relevant_docs.pkl', 'wb') as f:
            pickle.dump(dict, f)
    

def add_to_relevant_docs(key, value):
    with open('Videogame-ir-searchengine/relevant_docs.pkl', 'rb') as f:
        docs = pickle.load(f)
    if key not in docs:
        raise KeyError(f"KEY IS NOT IN HERE, Valid keys are: {', '.join(docs.keys())}")
    doc = nlp(value.lower())  
    docs[key].append(doc)
    with open('Videogame-ir-searchengine/relevant_docs.pkl', 'wb') as f:
        pickle.dump(docs, f)


    

def save_new_game_enties(dict):
    if dict:
        with open('Videogame-ir-searchengine/processed_game_entities.pkl', 'wb') as f:
            pickle.dump(dict, f)



