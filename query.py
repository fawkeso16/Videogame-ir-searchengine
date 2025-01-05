from processing import calculate_query_weights, get_total_documents, normalize_query_weights, tokenize_query
from nltk import PorterStemmer
import spacy
import pickle
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")

keywords = ["title", "genre", "developer", "publisher", "release date", "rating"]
stemmer = PorterStemmer()
stemmed_keywords = [stemmer.stem(word) for word in keywords]
unwanted_words = ["game", "games", "ps2"]

with open('processed_game_entites.pkl', 'rb') as f:
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

def process_query(query):
    tokenised_query = tokenize_query(query)
    hyphenated_words = [word.lower() for word in query.split() if '-' in word]
    filtered_query = [
        stemmer.stem(term) if term.endswith('s') else term
        for term in tokenised_query
        if term.lower() not in unwanted_words and
           term.lower() not in keywords and
           stemmer.stem(term.lower()) not in stemmed_keywords
    ] + hyphenated_words

    doc = nlp(" ".join(filtered_query))
    entity_matches = {ent.text.lower(): ent.text for ent in doc.ents if ent.label_ in processed_game_entities}

    for entity_key in entity_matches.keys():
        if entity_key not in filtered_query:
            filtered_query.append(entity_key)

        entity_terms = entity_key.split()
        for term in entity_terms:
            if term not in filtered_query:
                filtered_query.append(term)

    return filtered_query

def search_query(query):
    print(f"Original Query: {query}")
    
    filtered_query = process_query(query)
    print(f"Filtered Query: {filtered_query}")

    query_weights = calculate_query_weights(filtered_query, processed_game_entities)
    final_weights = normalize_query_weights(query_weights)

    return final_weights
