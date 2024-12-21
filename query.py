from processing import calculate_query_weights, get_total_documents, normalize_query_weights

def search_query(query):

    doc_count = get_total_documents()
    query_weights = calculate_query_weights(doc_count, query)
    final_weights = normalize_query_weights(query_weights)

    return final_weights
