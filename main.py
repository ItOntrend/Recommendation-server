import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from google.cloud.firestore_v1 import FieldFilter
import firebase_admin
from firebase_admin import credentials, firestore
import os

# ✅ Register the custom model for loading
@tf.keras.utils.register_keras_serializable()
class RecommenderModel(tf.keras.Model):
    def __init__(self, num_users=1, num_items=1, embedding_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        return tf.reduce_sum(user_vector * item_vector, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Initialize Firebase
cred = credentials.Certificate("firebaseKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load encoders and model
user_encoder = joblib.load("user_encoder.pkl")
item_encoder = joblib.load("item_encoder.pkl")
model = tf.keras.models.load_model("model.keras")

# Load orders for context
orders = db.collection('orders').stream()
order_data = []
for doc in orders:
    data = doc.to_dict()
    user_id = data.get("userID")
    items = data.get("items", [])
    for item in items:
        item_name = item.get("itemName")
        if user_id and item_name:
            order_data.append({"userID": user_id, "itemName": item_name})

df = pd.DataFrame(order_data)
# Filter to only known userIDs and itemNames
df = df[
    df['userID'].isin(user_encoder.classes_) &
    df['itemName'].isin(item_encoder.classes_)
]

# Encode
df['user'] = user_encoder.transform(df['userID'])
df['item'] = item_encoder.transform(df['itemName'])

# Flask app
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('userID')

    if user_id not in df['userID'].values:
        return jsonify({"success": False, "message": "User not found", "recommended_products": []})

    encoded_user = user_encoder.transform([user_id])[0]
    user_items = df[df['user'] == encoded_user]['item'].tolist()
    all_items = np.arange(len(item_encoder.classes_))
    items_to_predict = np.setdiff1d(all_items, user_items)

    user_item_pairs = np.array([[encoded_user, item] for item in items_to_predict])
    predicted_scores = model.predict(user_item_pairs)
    top_indices = predicted_scores.argsort()[-5:][::-1]
    recommended_item_ids = [items_to_predict[i] for i in top_indices]
    recommended_item_names = item_encoder.inverse_transform(recommended_item_ids)

    recommended_products = []
    for name in recommended_item_names:
        docs = db.collection_group('details').where(
            filter=FieldFilter("name", "==", name)
        ).limit(1).stream()
        for doc in docs:
            product_data = doc.to_dict()
            product_data['id'] = doc.id
            for key, value in list(product_data.items()):
                if isinstance(value, firestore.DocumentReference):
                    product_data[key] = str(value.path)
            recommended_products.append(product_data)

    return jsonify({
        "success": True,
        "userID": user_id,
        "recommended_products": recommended_products
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
