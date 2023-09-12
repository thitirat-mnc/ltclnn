import math

def rank_retrieve(query, inv_index, get_tfidf):
    """
    Given a query (a list of words), return a rank-ordered list of
    documents (by ID) and score for the query.
    """
    scores = [0.0] * len(inv_index)  # Initialize scores with zeros

    # Calculate the TF-IDF weighted scores
    for word in query:
        query_tf = 1 + math.log10(query.count(word))
        for d in inv_index.get(word, []):
            scores[d] += query_tf * get_tfidf(word, d)

    # Calculate the denominator for cosine normalization
    c = [0.0] * len(inv_index)
    for word in inv_index:
        for d in inv_index[word]:
            c[d] += get_tfidf(word, d) ** 2

    # Normalize the scores using cosine normalization
    for d in range(len(scores)):
        if scores[d] > 0:
            scores[d] /= math.sqrt(c[d])

    return scores

# Example usage:
# Replace the following with your actual data and functions
query = []  # List of query terms
inv_index = {}  # Inverted index mapping words to document IDs
get_tfidf = lambda word, document: 0.0  # Replace with your get_tfidf function

scores = rank_retrieve(query, inv_index, get_tfidf)
print("Document Scores:", scores)
