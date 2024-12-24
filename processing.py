import spacy
import math
import pandas as pd

nlp = spacy.load("en_core_web_sm")
inverted_index_dict = {}
metadata_dict = {}  
weights = {}
unique_terms = set()  


TITLE_BOOST = 6
GENRE_BOOST = 5
PUBLISHER_BOOST = 5
DEVELOPER_BOOST = 5
RATING_BOOST = 5
RELEASE_DATE_BOOST = 5



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

# Useful methods end 

def tokenise_texts(allFiles, metadata_dict):
    texts = list(allFiles.values())  
    file_names = list(allFiles.keys())  

    texts = [' '.join(text) if isinstance(text, list) else text for text in texts]

    docs = nlp.pipe(texts)

    for doc, file_name in zip(docs, file_names):
        filtered_tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_punct and not token.is_space and not token.is_stop
        ]

        # Add to the inverted index with metadata
        addToInvertedIndex(filtered_tokens, file_name, metadata_dict[file_name])


def tokenise_query(query):
    doc = nlp(query)
    filtered_tokens = [token.lemma_.lower() for token in doc 
                       if not token.is_punct and not token.is_space 
                       and not token.is_stop]
    return filtered_tokens

# Calculating tf-idf weight and normalizing methods
def calculate_document_weights(count):
    weights = {}  
    doc_count = count  

    for token, term_data in inverted_index_dict.items():
        doc_frequency = term_data["doc_frequency"]
        
        df = len(doc_frequency)
        if df == 0:
            continue  
        
        idf = math.log(1 + (doc_count / (1 + df)))
    
        for doc, tf in doc_frequency.items():
            term_frequency = math.log(1 + tf["tf"]) / term_data["total_tf"]
            tf_idf = term_frequency * idf

            if token in metadata_dict[doc]["title"]:  # Assuming title is part of metadata_dict
                tf_idf *= TITLE_BOOST  # Increase the weight if it's in the title
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
        tf = math.log(1 + query_tokens.count(term)) 
        idf = get_doc_count_term_frequency(term)  
        inverse_document_frequency = math.log(count / idf) if idf > 0 else 0

        weight = tf * inverse_document_frequency

        if term in [metadata["title"] for metadata in metadata_dict.values()]:
            weight *= TITLE_BOOST  # Increase the weight if it's in the title
        elif term in [metadata["genre"] for metadata in metadata_dict.values()]:
            weight *= GENRE_BOOST
        elif term in [metadata["developer"] for metadata in metadata_dict.values()]:
            weight *= DEVELOPER_BOOST
        elif term in[metadata["publisher"] for metadata in metadata_dict.values()]:
            weight *= PUBLISHER_BOOST
        elif term in [metadata["rating"] for metadata in metadata_dict.values()]:
            weight *= RATING_BOOST
        elif term in[metadata["releaseDate"] for metadata in metadata_dict.values()]:
            weight *= RELEASE_DATE_BOOST


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
        
        dot_product = sum(query_weights.get(term, 0) * term_weights.get(term, 0) for term in query_weights)

        if query_magnitude != 0 and doc_magnitude != 0:
            cosine_similarities[doc_id] = dot_product / (query_magnitude * doc_magnitude)
        else:
            cosine_similarities[doc_id] = 0.0 

    best_doc_id = max(cosine_similarities, key=cosine_similarities.get, default=None)
    best_similarity = cosine_similarities.get(best_doc_id, 0.0)
    
    return best_doc_id, best_similarity

# Getters for testing 
def get_unique_terms():
    return unique_terms

def getInvertedIndex():
    return inverted_index_dict

def getMetadata():
    return metadata_dict
