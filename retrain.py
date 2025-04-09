import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate("firebaseKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Step 1: Fetch orders from Firestore
print("Fetching orders from Firestore...")
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

# Step 2: Create DataFrame
df = pd.DataFrame(order_data)

if df.empty:
    raise ValueError("No order data found to retrain model.")

# Step 3: Encode userID and itemName
print("Encoding data...")
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user'] = user_encoder.fit_transform(df['userID'])
df['item'] = item_encoder.fit_transform(df['itemName'])

num_users = df['user'].nunique()
num_items = df['item'].nunique()

# Step 4: Prepare training data
train_data = df[['user', 'item']].values
ratings = tf.ones(len(train_data))  # implicit feedback

# Step 5: Define and train model
print("Training recommendation model...")
@tf.keras.utils.register_keras_serializable()
class RecommenderModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super().__init__()
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
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = RecommenderModel(num_users, num_items)
model.compile(optimizer='adam', loss='mse')
model.fit(train_data, ratings, epochs=9)

# 🛠️ Force the model to build before saving
model(tf.constant([[0, 0]]))  # Important!

# Step 6: Save model and encoders
print("Saving model and encoders...")
model.save("model.keras")
joblib.dump(user_encoder, "user_encoder.pkl")
joblib.dump(item_encoder, "item_encoder.pkl")

print("✅ Retraining complete and saved successfully.")
