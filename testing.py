from processing import compute_cosine_similarity, precision_and_recall, final_results
from query import search_query
import matplotlib.pyplot as plt


#cosine graphing.
def cosine_graph(data_dict):

    queries = list(data_dict.keys())
    scores = [item[1] for item in data_dict.values()]
    plt.figure(figsize=(10, 6))
    plt.plot(queries, scores, marker='o', linestyle='-', color='b', label="Cosine Similarity")
    plt.xticks(rotation=45, ha='right')  
    plt.xlabel("Test Queries")
    plt.ylabel("Cosine Similarity Score")
    plt.title("Cosine Similarity Scores for Test Queries")
    plt.ylim(0, 1)  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def precision_recall_graph(data_dict, relevant_docs, weights):

    plt.figure(figsize=(10, 6))

    for query_key, query_value in data_dict.items():
        print(f"Processing Query: {query_key} - {query_value}") 
        relevant_docs_for_query = relevant_docs[query_key]    
        weighted_query = search_query(query_value)
        # print(weighted_query)
        retrieved_docs = compute_cosine_similarity(weighted_query, weights)
        
        # print(f"Relevant Docs for {query_key}: {relevant_docs_for_query}")  
        # print(f"Retrieved Docs for {query_key}: {retrieved_docs}...")  

        precision , recall = precision_and_recall(retrieved_docs, relevant_docs_for_query)  
        
        # print(f"Precision: {precision:}, Recall: {recall},") 
        label_text = f"{query_key}: Precision={precision}, Recall={recall}"  
        plt.scatter(recall, precision, marker="o", label=label_text)
        print(final_results(retrieved_docs))


    plt.title("Precision-Recall Points for All Queries")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')

    plt.tight_layout()  
    plt.show()
