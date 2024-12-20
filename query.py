from processing import tokenise_texts, calculate_query_weights, tokenise_query, get_total_documents, normalize_query_weights

def search_query(query):

    # Get the total document count (to use in IDF calculation)
    doc_count = get_total_documents()

    # Calculate query weights
    query_weights = calculate_query_weights(doc_count, query)

    final_weights = normalize_query_weights(query_weights)

    # Output the query weights (or do further search logic here)
    # print(query_weights)
    # print(final_weights)

    return final_weights
