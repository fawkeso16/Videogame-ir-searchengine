from processing import calculate_query_weights, get_total_documents, normalize_query_weights, tokenize_query
import nltk
import spacy
import pickle
from spacy.pipeline import EntityRuler


nlp = spacy.load("en_core_web_sm")

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


import requests

def get_synonyms(word):
    url = f"https://api.datamuse.com/words?rel_syn={word}"
    response = requests.get(url)
    if response.status_code == 200:
        return [item['word'] for item in response.json()]
    return []

def process_query(query):
    allsynonyms = []
    tokenised_query = tokenize_query(query)
    hyphenated_words = [word.lower() for word in query.split() if '-' in word]
    filtered_query = tokenised_query + hyphenated_words

    doc = nlp(" ".join(filtered_query))
    entity_matches = {ent.text.lower(): ent.text for ent in doc.ents if ent.label_ in processed_game_entities}

    for entity_key in entity_matches.keys():
        if entity_key not in filtered_query:
            filtered_query.append(entity_key)

        entity_terms = entity_key.split()
        for term in entity_terms:
            if term not in filtered_query:
                filtered_query.append(term)

    # Apply thesaurus-based query expansion
    expanded_query = filtered_query.copy()
    for token in filtered_query:
        synonyms = get_synonyms(token)
        for synonym in synonyms:
            if synonym not in expanded_query:
                expanded_query.append(synonym)
                allsynonyms.append(synonym)
                # print('SYNONYM ADDED: ', synonym)

    return expanded_query, allsynonyms


def search_query(query):
    print(f"Original Query: {query}")
    
    filtered_query, synonyms = process_query(query)
    print(f"Filtered Query: {filtered_query}")

    query_weights = calculate_query_weights(filtered_query, synonyms)
    final_weights = normalize_query_weights(query_weights)

    return final_weights
