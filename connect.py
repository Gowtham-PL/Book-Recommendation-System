import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB (Make sure MongoDB is running)
client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB connection string if needed
 
# Create (or connect to) a database
db = client["Book_recommendation"]

# Load the dataset from CSV
df = pd.read_csv("Books_Data_Clean.csv", encoding='latin1')
print(df.head())  # Replace with the actual file path

# Convert DataFrame to dictionary format for MongoDB
data_dict = df.to_dict(orient="records")

# Insert data into MongoDB collection
collection = db["Books"]
collection.insert_many(data_dict)

print("✅ Dataset successfully imported into MongoDB!")