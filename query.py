# Date: 16 December 2024
# Author: Oliver Fawkes
# Description: Contains methods for processing queries, expanding with synonyms and matching entities(NER). 
# History: 
#  16/12/2024 - v1.00 - Initial methods for processing queries, tokenizing, and matching entities.
#  17/12/2024 - v1.10 - Finished implementing entity matching using of spacy and only custom entitys
#  18/12/2024 - v1.20 - Added keyword macthing and removed unwnated words (ps2), game etc .
#  26/12/2024 - v1.30 - Bug fixing entity matching and query expansion code, changed synonym api.
#  28/12/2024 - v1.40 - Improved structure of query processing and entity handling, integrated new tokenization and query expansion strategy.
#  30/12/2024 - v1.50 - Refined entity matching for complex entities, removed unwanted words due to change in metadata structure.
#  07/01/2024 - v1.60 - Final bug fixes , improved handling of entity or better search results, also removed keyword matching (was irrelevant), cleaned code and added comments.


from processing import calculate_query_weights, normalize_query_weights, tokenize_query
import spacy
import requests
import config
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm", disable=["ner"])  
ruler = nlp.add_pipe("entity_ruler", before="ner")


processed_game_entities = config.processed_game_entities
patterns = config.patterns
ruler.add_patterns(patterns)


# Fetch synonyms for a word using the Datamuse API
def get_synonyms(word):
    url = f"https://api.datamuse.com/words?rel_syn={word}"
    response = requests.get(url)
    if response.status_code == 200:
        synonyms = [item['word'] for item in response.json()]    
        return synonyms[:2]
    return []

#Method to process query using entity matching and thesaurus query expansion
def process_query(query):
    all_synonyms = []
    entities = {}

    tokenized_query = tokenize_query(query)
    hyphenated_words = [word.lower() for word in query.split() if '-' in word]
    filtered_query = tokenized_query + hyphenated_words

    doc = nlp(" ".join(filtered_query))
    entity_matches = {ent.text.lower(): ent.text for ent in doc.ents if ent.label_ in processed_game_entities}
    
    for entity_key, original_entity in entity_matches.items():
        print(f"Entity matches: {entity_matches}")

        if entity_key not in entities:
            for ent in doc.ents:
                if ent.text.lower() == entity_key:
                    entities[ent.label_] = original_entity
                    break

        # Add the entity and its terms to filtered query
        if entity_key not in filtered_query:
            filtered_query.append(entity_key) 
        
        if len(entity_key.split()) > 1:  
            for term in entity_key.split():
                if term not in filtered_query:
                    filtered_query.append(term)
        else:
            if entity_key not in filtered_query:
                filtered_query.append(entity_key)

    # Synonym expansion
    expanded_query = filtered_query.copy()
    for token in filtered_query:
        synonyms = get_synonyms(token)
        for synonym in synonyms:
            if synonym not in expanded_query:
                expanded_query.append(synonym)
                all_synonyms.append(synonym)

    return expanded_query, entities, all_synonyms


def search_query(query):
    print(f"Original Query: {query}")
    filtered_query, entity_matches, syonyms = process_query(query)
    print(f"Filtered Query: {filtered_query}")
    query_weights = calculate_query_weights(filtered_query, entity_matches, syonyms)
    final_weights = normalize_query_weights(query_weights)
    print("final weights: ", final_weights)

    return final_weights

