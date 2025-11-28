from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import re
import numpy as np
from difflib import get_close_matches
import os

app = Flask(__name__)
CORS(app)

# File paths
PROCESSED_PATH = "processed_books_data.csv"
TFIDF_PATH = "tfidf_vectorizer_books.pkl"
COSINE_PATH = "cosine_similarity_books.pkl"

# Global variables
df = None
tfidf_vectorizer = None
cosine_sim = None

def load_models():
    global df, tfidf_vectorizer, cosine_sim

    try:
        df = pd.read_csv(PROCESSED_PATH)
        df.fillna("", inplace=True)

        with open(TFIDF_PATH, "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        with open(COSINE_PATH, "rb") as f:
            cosine_sim = pickle.load(f)

        print("✅ Models and data loaded successfully.")
        print("📂 Columns:", df.columns.tolist())
        return True
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False

def exact_genre_match(query):
    """Returns exact genre match only for specific genres"""
    query = query.lower()
    
    # Direct mapping for the three specific genres
    if "genrefiction" in query:
        return "genrefiction"
    elif "nonfiction" in query:
        return "nonfiction"
    elif "fiction" in query and "genrefiction" not in query:
        return "fiction"
    
    return None

def smart_book_recommend(query, top_n=5):
    global df
    if df is None:
        return []

    query = query.lower()
    
    # Check for exact genre match first (strict matching)
    exact_genre = exact_genre_match(query)

    # Extract max price
    max_price = np.inf
    price_match = re.search(r"under\s*\$?(\d+(?:\.\d+)?)", query)
    if price_match:
        max_price = float(price_match.group(1))

    # Extract min rating
    min_rating = 0.0
    rating_match = re.search(r"rating\s*(?:above|greater than|over)\s*(\d+(\.\d+)?)", query)
    if rating_match:
        min_rating = float(rating_match.group(1))

    # Extract author
    author_input = ""
    author_match = re.search(r"(?:by|written by)\s+([a-z\s]+)", query)
    if author_match:
        author_input = author_match.group(1).strip()

    filtered_df = df.copy()

    # Apply genre filter (exact match for specified genres)
    genre_col = 'Genre' if 'Genre' in df.columns else ('genre' if 'genre' in df.columns else None)
    if exact_genre and genre_col:
        print(f"📚 Filtering for exact genre: {exact_genre}")
        filtered_df = filtered_df[filtered_df[genre_col].str.lower() == exact_genre]

    # Apply author filter
    if author_input:
        all_authors = df["Author"].dropna().unique()
        closest = get_close_matches(author_input.lower(), [a.lower() for a in all_authors if isinstance(a, str)], n=1, cutoff=0.6)
        if closest:
            filtered_df = filtered_df[filtered_df["Author"].str.lower() == closest[0]]

    # Apply price filter
    price_col = 'Price' if 'Price' in df.columns else ('sale price' if 'sale price' in df.columns else None)
    if price_col and max_price != np.inf:
        filtered_df = filtered_df[filtered_df[price_col] <= max_price]

    # Apply rating filter
    rating_col = 'Rating' if 'Rating' in df.columns else ('Book_average_rating' if 'Book_average_rating' in df.columns else None)
    if rating_col and min_rating > 0.0:
        filtered_df = filtered_df[filtered_df[rating_col] >= min_rating]

    # Build results
    result = []
    for _, book in filtered_df.head(top_n).iterrows():
        book_data = {
            "title": book.get("Title", book.get("Book Name", "Unknown")),
            "author": book.get("Author", "Unknown Author"),
        }
        
        if price_col:
            try:
                price_val = float(book[price_col])
                book_data["price"] = f"${price_val:.2f}"
            except (ValueError, TypeError):
                book_data["price"] = "Price not available"
        else:
            book_data["price"] = "Price not available"
            
        if genre_col:
            book_data["genre"] = book.get(genre_col, "Uncategorized")
        else:
            book_data["genre"] = "Uncategorized"
            
        if rating_col:
            try:
                book_data["rating"] = float(book[rating_col])
            except (ValueError, TypeError):
                book_data["rating"] = 0.0
        else:
            book_data["rating"] = 0.0
            
        result.append(book_data)

    return result

def content_based_recommend(book_title, top_n=5):
    """Content-based book recommendation with similarity scores"""
    global df, cosine_sim

    if df is None or cosine_sim is None:
        return []

    # Find title column
    title_col = 'Title' if 'Title' in df.columns else ('Book Name' if 'Book Name' in df.columns else None)
    if not title_col:
        return []
        
    # Create indices series for fast lookup
    indices = pd.Series(df.index, index=df[title_col]).drop_duplicates()
    
    # Handle case where exact title isn't found
    if book_title not in indices:
        closest = get_close_matches(book_title, indices.index, n=1, cutoff=0.5)
        if not closest:
            return []
        book_title = closest[0]
        print(f"📖 Using closest match: {book_title}")

    # Get index of the book
    idx = indices[book_title]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Remove the book itself
    if sim_scores[0][0] == idx:
        sim_scores = sim_scores[1:]
        
    # Get top N
    sim_scores = sim_scores[:top_n]
    book_indices = [i[0] for i in sim_scores]

    # Build result with similarity scores
    result = []
    for i, score in enumerate(sim_scores):
        book_idx = score[0]
        similarity = score[1]
        book = df.iloc[book_idx]
        
        book_data = {
            "title": book[title_col],
            "author": book["Author"],
            "similarity": f"{similarity:.2f}"  # Include similarity score
        }
        
        # Add genre if available
        genre_col = 'Genre' if 'Genre' in df.columns else ('genre' if 'genre' in df.columns else None)
        if genre_col:
            book_data["genre"] = book[genre_col]
        
        # Add rating if available
        rating_col = 'Rating' if 'Rating' in df.columns else ('Book_average_rating' if 'Book_average_rating' in df.columns else None)
        if rating_col:
            try:
                book_data["rating"] = float(book[rating_col])
            except (ValueError, TypeError):
                book_data["rating"] = 0.0
            
        # Add price if available
        price_col = 'Price' if 'Price' in df.columns else ('sale price' if 'sale price' in df.columns else None)
        if price_col:
            try:
                price_val = float(book[price_col])
                book_data["price"] = f"${price_val:.2f}"
            except (ValueError, TypeError):
                book_data["price"] = "Price not available"
        
        result.append(book_data)
        
    return result

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/<page>')
def serve_page(page):
    try:
        return render_template(page)
    except:
        return "❌ Page not found", 404

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        query = data.get("query", "")
        print("📥 Query received:", query)

        if not query:
            return jsonify({"recommendations": []})
            
        # Determine if this is likely a book title query or a filtering query
        is_title_query = not any(keyword in query.lower() for keyword in 
                             ["under", "above", "between", "rating", "price", "by", "written by", 
                              "genrefiction", "fiction", "nonfiction"])
        
        if is_title_query and len(query.split()) > 1:
            print("📚 Processing as book title query")
            results = content_based_recommend(query)
        else:
            print("🔍 Processing as filter query")
            results = smart_book_recommend(query)

        if not results:
            return jsonify({"recommendations": [], "message": "No results found."})

        return jsonify({"recommendations": results})
    except Exception as e:
        print(f"❌ Error in /recommend: {str(e)}")
        return jsonify({"error": str(e), "message": "Something went wrong with the recommendation system. Please try again later."}), 500

# Initialize models when app starts
load_models()

if __name__ == '__main__':
    app.run(debug=True)