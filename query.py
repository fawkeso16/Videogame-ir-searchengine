from processing import calculate_query_weights, get_total_documents, normalize_query_weights, tokenize_query
from nltk import PorterStemmer
import spacy

# Load the spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Define unwanted words and keywords
keywords = ["title", "genre", "developer", "publisher", "release date", "rating"]
stemmer = PorterStemmer()
stemmed_keywords = [stemmer.stem(word) for word in keywords]
unwanted_words = ["game", "games", "ps2"]

def process_query(query):
    # Split the query by spaces and check for hyphenated words
    hyphenated_words = [word for word in query.split() if '-' in word]
    
    # Tokenize the query and filter out unwanted words, lemmatize, etc.
    tokenised_query = tokenize_query(query)

    # Re-add hyphenated words after tokenization
    filtered_query = []
    for term in tokenised_query:
        if term.lower() in unwanted_words:
            continue
        if term.lower() in keywords or stemmer.stem(term.lower()) in stemmed_keywords:
            continue
        filtered_query.append(term)

    for hyphenated_word in hyphenated_words:
        filtered_query.append(hyphenated_word.lower())

    return filtered_query

def search_query(query, game_entities, all_titles):
    print(f"Original Query: {query}")    
    filtered_query = process_query(query)
    print(f"Filtered Query: {filtered_query}")
    entity_matches = []
    matching_keywords = []

    # Iterate through the filtered query to find entity matches
    for term in filtered_query:
        for entity_type, values in game_entities.items():
            if any(term.lower() == value.lower() for value in values):
                entity_matches.append((term, entity_type))
                break

    print(f"Entity Matches: {entity_matches}")

    for title in all_titles:
        if title.lower() in ' '.join(filtered_query).lower():
            entity_matches.append((title, "TITLE"))
    
    print(f"Updated Entity Matches: {entity_matches}")

    query_weights = calculate_query_weights(filtered_query, entity_matches, [])
    final_weights = normalize_query_weights(query_weights)

    return final_weights
