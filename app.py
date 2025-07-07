from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "./knn_model.pkl"
MATRIX_PATH = "./user_item_matrix.pkl"
INDEX_MAP_PATH = "./customer_index_map.pkl"
PRODUCTS_PATH = "./olist_products_dataset.csv"
CATEGORY_PATH = "./product_category_name_translation.csv"

missing_files = []
for path in [MODEL_PATH, MATRIX_PATH, INDEX_MAP_PATH, PRODUCTS_PATH, CATEGORY_PATH]:
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    raise FileNotFoundError(f"Required files not found: {', '.join(missing_files)}")

with open(MODEL_PATH, "rb") as f:
    knn_model = pickle.load(f)

user_item_matrix = pd.read_pickle(MATRIX_PATH)

with open(INDEX_MAP_PATH, "rb") as f:
    customer_index_map = pickle.load(f)

products_df = pd.read_csv(PRODUCTS_PATH)
category_df = pd.read_csv(CATEGORY_PATH)
products_with_category = pd.merge(products_df, category_df, on='product_category_name', how='left')

def get_recommendations_fixed(customer_index, n_neighbors=5, top_n=5):
    distances, indices = knn_model.kneighbors(user_item_matrix.iloc[customer_index:customer_index+1],
                                              n_neighbors=n_neighbors+1)
    similar_users = indices.flatten()[1:]
    similar_users_matrix = user_item_matrix.iloc[similar_users]
    product_scores = similar_users_matrix.sum(axis=0)

    user_purchases = user_item_matrix.iloc[customer_index]
    purchased_products = user_purchases[user_purchases > 0].index

    product_scores = product_scores.drop(purchased_products, errors='ignore')
    top_products = product_scores.sort_values(ascending=False).head(top_n).index.tolist()

    recommendations = products_with_category[products_with_category['product_id'].isin(top_products)]
    return recommendations[['product_id', 'product_category_name_english']].drop_duplicates().values.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    customer_id = request.form['customer_id']

    if customer_id not in customer_index_map['id_to_index']:
        return render_template('result.html', recommendations=[], error="Customer ID not found.")

    index = customer_index_map['id_to_index'][customer_id]
    recommendations = get_recommendations_fixed(index)

    return render_template('result.html', recommendations=recommendations, error=None)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
