from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util
import sqlite3
import time

api_blueprint = Blueprint('api', __name__)

# Load Sentence Transformer model for semantic search
semantic_search_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to SQLite database
db_connection = sqlite3.connect('residentialDatabase.db')
db_cursor = db_connection.cursor()

# Timer and log for database query
start_time = time.time()
db_cursor.execute("SELECT * FROM residentialDatabase WHERE MLS IS NOT NULL")
rows = db_cursor.fetchall()
query_time = time.time() - start_time
print(f'Database query time: {query_time} seconds')

# Log the number of rows fetched
num_rows = len(rows)
print(f'Number of rows fetched: {num_rows}')

# Get column names from the cursor description
column_names = [desc[0] for desc in db_cursor.description]

# Timer and log for encoding
start_time = time.time()
encoded_passages = []
mls_values = []  # New array to store MLS values
for row in rows:
    passage = ""
    mls_value = None  # Initialize MLS value as None
    for i, value in enumerate(row):
        column_name = column_names[i]
        if value is not None:
            passage += f"{column_name}: {value}. "
            if column_name == 'MLS':
                mls_value = value  # Store MLS value
    encoded_passages.append(passage)
    mls_values.append(mls_value)  # Append MLS value
encoding_time = time.time() - start_time
print(f'Encoding time: {encoding_time} seconds')

# Encode property details
start_time = time.time()
corpus_embeddings = semantic_search_model.encode(encoded_passages)
embedding_time = time.time() - start_time
print(f'Embedding time: {embedding_time} seconds')

@api_blueprint.route('/query', methods=['POST'])
def process_query():
    try:
        # Get the query from the JSON payload
        query_data = request.get_json()
        user_query = query_data['query']

        # Encode the user query
        start_time = time.time()
        query_embedding = semantic_search_model.encode(user_query, convert_to_tensor=True)
        query_encoding_time = time.time() - start_time
        print(f'Query encoding time: {query_encoding_time} seconds')

        # Perform semantic search
        start_time = time.time()
        hits = util.semantic_search(query_embedding.unsqueeze(0), corpus_embeddings, top_k=10)
        search_time = time.time() - start_time
        print(f'Semantic search time: {search_time} seconds')

        hits = hits[0]  # Get the hits for the current query

        # Prepare the response
        response_data = {
            'user_query': user_query,
            'results': [
                {
                    'passage': encoded_passages[hit['corpus_id']],
                    'mls_value': mls_values[hit['corpus_id']],
                    'score': hit['score']
                }
                for hit in hits
            ]
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
