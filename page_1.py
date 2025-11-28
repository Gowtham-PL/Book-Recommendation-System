import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# File paths
DATA_PATH = "Books_Data_Clean.csv"
TFIDF_PATH = "tfidf_vectorizer_books.pkl"
COSINE_PATH = "cosine_similarity_books.pkl"
PROCESSED_PATH = "processed_books_data.csv"

# Global variables
df = None
tfidf_vectorizer = None
cosine_sim = None

def preprocess_and_train():
    global df, tfidf_vectorizer, cosine_sim

    # Load raw data
    df = pd.read_csv(DATA_PATH)
    
    # Print column names for verification
    print("Original columns:", df.columns.tolist())
    
    # Make copies of original columns to ensure consistency
    column_mapping = {
        'Book Name': 'Title',
        'Book_average_rating': 'Rating',
        'genre': 'Genre',
        'sale price': 'Price'
    }
    
    # Apply mapping for columns that exist
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
                
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Keep relevant columns and clean
    required_cols = ['Title', 'Author', 'Author_Rating', 'Rating', 'Genre', 'Price']
    available_cols = [col for col in required_cols if col in df.columns]
    
    df = df[available_cols].copy()
    df.fillna("", inplace=True)
    
    # Create combined features for TF-IDF
    feature_cols = [col for col in ['Title', 'Author', 'Genre'] if col in df.columns]
    df['combined'] = df[feature_cols].apply(lambda row: ' '.join(str(value) for value in row), axis=1)

    # Train TF-IDF model
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined'])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Evaluate model accuracy
    evaluate_model_accuracy()

    # Save processed data and models
    df.to_csv(PROCESSED_PATH, index=False)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(COSINE_PATH, "wb") as f:
        pickle.dump(cosine_sim, f)

    print("✅ Model trained and saved successfully!")

def evaluate_model_accuracy():
    """Evaluate the recommendation model accuracy"""
    global df, cosine_sim
    
    if 'Genre' not in df.columns:
        genre_col = 'genre' if 'genre' in df.columns else None
        if genre_col:
            df['Genre'] = df[genre_col]
        else:
            print("❌ Cannot evaluate accuracy: No genre column found")
            return
    
    # Create test cases by splitting data
    if len(df) > 100:  # Only evaluate if we have enough data
        test_size = min(100, int(len(df) * 0.2))
        test_indices = np.random.choice(df.index, size=test_size, replace=False)
        test_books = df.loc[test_indices]
        
        correct_predictions = 0
        total_predictions = 0
        
        # For each test book, see if we recommend books of the same genre
        for idx, test_book in test_books.iterrows():
            title = test_book['Title']
            true_genre = test_book['Genre'].lower()
            
            # Skip if no genre info
            if not true_genre:
                continue
                
            # Get recommendations
            recommendations = content_based_recommend(title, top_n=5, exclude_self=True)
            if not recommendations:
                continue
                
            # Check if recommended books have the same genre
            for rec in recommendations:
                total_predictions += 1
                rec_genre = rec['genre'].lower()
                if rec_genre == true_genre:
                    correct_predictions += 1
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"\n📊 MODEL EVALUATION:")
            print(f"Total test predictions: {total_predictions}")
            print(f"Correct genre predictions: {correct_predictions}")
            print(f"Model accuracy: {accuracy:.2%}")
            print(f"This means {accuracy:.2%} of recommendations have matching genres to the input book.")
        else:
            print("⚠️ Not enough data for meaningful accuracy evaluation")

def initialize():
    global df, tfidf_vectorizer, cosine_sim
    try:
        df = pd.read_csv(PROCESSED_PATH)
        with open(TFIDF_PATH, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open(COSINE_PATH, "rb") as f:
            cosine_sim = pickle.load(f)
        print("📚 Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
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

def extract_rating_range(query):
    between_match = re.search(r'rating\s*(?:is)?\s*between\s*(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*(\d+(?:\.\d+)?)', query)
    above_match = re.search(r'rating\s*(?:is)?\s*above\s*(\d+(?:\.\d+)?)', query)
    below_match = re.search(r'rating\s*(?:is)?\s*below\s*(\d+(?:\.\d+)?)', query)

    if between_match:
        return float(between_match.group(1)), float(between_match.group(2))
    elif above_match:
        return float(above_match.group(1)), np.inf
    elif below_match:
        return 0.0, float(below_match.group(1))
    return 0.0, np.inf

def extract_price_range(query):
    # Enhanced patterns with dollar sign variations
    between_match = re.search(r'price\s*(?:is)?\s*between\s*\$?(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?(\d+(?:\.\d+)?)', query)
    above_match = re.search(r'(?:price\s*(?:is)?\s*above|above\s*\$?|over\s*\$?)(\d+(?:\.\d+)?)', query)
    below_match = re.search(r'(?:price\s*(?:is)?\s*below|below\s*\$?)(\d+(?:\.\d+)?)', query)
    under_match = re.search(r'under\s*\$?(\d+(?:\.\d+)?)', query)
    
    # Direct dollar amount pattern
    dollar_range = re.search(r'\$(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?(\d+(?:\.\d+)?)', query)

    if between_match:
        print(f"Found price between: {between_match.group(1)} and {between_match.group(2)}")
        return float(between_match.group(1)), float(between_match.group(2))
    elif dollar_range:
        print(f"Found dollar range: {dollar_range.group(1)} to {dollar_range.group(2)}")
        return float(dollar_range.group(1)), float(dollar_range.group(2))
    elif above_match:
        print(f"Found price above: {above_match.group(1)}")
        return float(above_match.group(1)), np.inf
    elif below_match or under_match:
        match = below_match or under_match
        print(f"Found price below: {match.group(1)}")
        return 0.0, float(match.group(1))
    return 0.0, np.inf

def smart_book_recommend(query, top_n=10):
    """Advanced book recommendation based on natural language query"""
    global df

    if df is None:
        print("❌ Dataset not loaded!")
        return []

    query = query.lower()
    
    # Check for exact genre match first
    exact_genre = exact_genre_match(query)
    
    # Extract other filters
    rating_min, rating_max = extract_rating_range(query)
    price_min, price_max = extract_price_range(query)

    # Extract author
    author_match = re.search(r"(?:by|written by)\s+([a-z\s]+)", query)
    author_input = author_match.group(1).strip() if author_match else ""

    print(f"🔍 Parsed - Genre: '{exact_genre}', Rating: {rating_min}-{rating_max}, Price: {price_min}-{price_max}, Author: '{author_input}'")

    # Start with full dataset
    filtered_df = df.copy()

    # Apply genre filter if specified
    genre_col = 'Genre' if 'Genre' in df.columns else ('genre' if 'genre' in df.columns else None)
    if exact_genre and genre_col:
        print(f"Filtering for exact genre: {exact_genre}")
        # Use exact string match for the specified genres
        filtered_df = filtered_df[filtered_df[genre_col].str.lower() == exact_genre]
        if filtered_df.empty:
            print(f"😕 No books found for genre: {exact_genre}")
            return []

    # Apply author filter
    if author_input:
        all_authors = df['Author'].dropna().unique()
        closest = get_close_matches(author_input.lower(), [a.lower() for a in all_authors if isinstance(a, str)], n=1, cutoff=0.6)
        if closest:
            filtered_df = filtered_df[filtered_df['Author'].str.lower() == closest[0]]
        else:
            return []

    # Apply rating filter
    rating_col = 'Rating' if 'Rating' in df.columns else ('Book_average_rating' if 'Book_average_rating' in df.columns else None)
    if rating_col:
        filtered_df = filtered_df[
            (filtered_df[rating_col] >= rating_min) &
            (filtered_df[rating_col] <= rating_max)
        ]

    # Apply price filter - FIXED VERSION
    price_col = 'Price' if 'Price' in df.columns else ('sale price' if 'sale price' in df.columns else None)
    if price_col and (price_min > 0 or price_max < np.inf):
        # Make sure price column is numeric
        try:
            # First remove any currency symbols and convert to float
            if filtered_df[price_col].dtype == 'object':
                filtered_df[price_col] = filtered_df[price_col].astype(str).str.replace('[$,]', '', regex=True)
                filtered_df[price_col] = pd.to_numeric(filtered_df[price_col], errors='coerce')
            
            # Apply price filter with strict enforcement
            price_mask = (filtered_df[price_col] >= price_min) & (filtered_df[price_col] <= price_max)
            filtered_df = filtered_df[price_mask]
            
            print(f"Price filter applied: ${price_min} to ${price_max}, remaining books: {len(filtered_df)}")
            
            # Debug: show a sample of filtered prices to verify
            if len(filtered_df) > 0:
                sample_prices = filtered_df[price_col].head(5).tolist()
                print(f"Sample prices after filtering: {sample_prices}")
            
        except Exception as e:
            print(f"Error in price filtering: {e}")
    
    if filtered_df.empty:
        print("No books found matching all criteria.")
        return []

    # Build result
    result = []
    for _, book in filtered_df.head(top_n).iterrows():
        book_data = {
            "title": book.get("Title", book.get("Book Name", "Unknown")),
            "author": book.get("Author", "Unknown"),
        }
        
        if 'Author_Rating' in book:
            book_data["author_rating"] = book["Author_Rating"]
            
        if rating_col:
            book_data["rating"] = book[rating_col]
            
        if price_col:
            book_data["price"] = book[price_col]
            
        if genre_col:
            book_data["genre"] = book[genre_col]
            
        result.append(book_data)

    return result

def content_based_recommend(book_title, top_n=10, exclude_self=False):
    """Content-based book recommendation with similarity scores"""
    global df, cosine_sim

    if df is None or cosine_sim is None:
        print("❌ Models not loaded")
        return []

    # Find title column
    title_col = 'Title' if 'Title' in df.columns else ('Book Name' if 'Book Name' in df.columns else None)
    if not title_col:
        print("❌ No title column found in dataset")
        return []
        
    # Create indices series for fast lookup
    indices = pd.Series(df.index, index=df[title_col]).drop_duplicates()
    
    # Handle case where exact title isn't found
    if book_title not in indices:
        closest = get_close_matches(book_title, indices.index, n=1, cutoff=0.5)
        if not closest:
            print(f"❌ No match found for book title: {book_title}")
            return []
        book_title = closest[0]
        print(f"📖 Using closest match: {book_title}")

    # Get index of the book
    idx = indices[book_title]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Remove the book itself if requested
    if exclude_self:
        sim_scores = sim_scores[1:]
    elif sim_scores[0][0] == idx:
        # Always remove the exact same book
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
            "similarity": f"{similarity:.2f}"  # Format to 2 decimal places
        }
        
        # Add genre if available
        genre_col = 'Genre' if 'Genre' in df.columns else ('genre' if 'genre' in df.columns else None)
        if genre_col:
            book_data["genre"] = book[genre_col]
        
        # Add rating if available
        rating_col = 'Rating' if 'Rating' in df.columns else ('Book_average_rating' if 'Book_average_rating' in df.columns else None)
        if rating_col:
            book_data["rating"] = book[rating_col]
            
        # Add price if available
        price_col = 'Price' if 'Price' in df.columns else ('sale price' if 'sale price' in df.columns else None)
        if price_col:
            book_data["price"] = book[price_col]
        
        result.append(book_data)
        
    return result

# Run training if needed or initialize existing models
if not os.path.exists(PROCESSED_PATH) or not os.path.exists(TFIDF_PATH) or not os.path.exists(COSINE_PATH):
    print("🔄 Training new model...")
    preprocess_and_train()
else:
    initialize()
    print("💡 TIP: To retrain the model and see accuracy metrics, delete the processed_books_data.csv file")

# Interactive usage
if __name__ == "__main__":
    if not df is None:  # Only run if we have data
        print("\n📚 BOOK RECOMMENDATION SYSTEM")
        print("=" * 50)
        print("Options:")
        print("1. Enter a book title to get similar recommendations")
        print("2. Enter a query like 'fiction books under $15'")
        print("3. Type 'genrefiction', 'fiction', or 'nonfiction' for genre-specific recommendations")
        print("4. Type 'exit' to quit")
        print("=" * 50)
        
        while True:
            user_input = input("\n📥 Enter your query:\n> ").strip()
            
            if user_input.lower() == "exit":
                print("👋 See you next time!")
                break
                
            # Check if it's likely a book title (no special keywords)
            is_title_query = not any(keyword in user_input.lower() for keyword in 
                                 ["under", "above", "between", "rating", "price", "by", "written by", 
                                  "genrefiction", "fiction", "nonfiction"])
            
            if is_title_query and len(user_input.split()) > 1:
                print(f"🔍 Searching for books similar to: {user_input}")
                recommendations = content_based_recommend(user_input)
                if recommendations:
                    print("\n📘 Similar Books:")
                    for idx, r in enumerate(recommendations, 1):
                        sim_text = f"[Similarity: {r['similarity']}]"
                        print(f"{idx}. {r['title']} by {r['author']} {sim_text}")
                        if 'genre' in r:
                            print(f"   Genre: {r['genre']}")
                        if 'rating' in r:
                            print(f"   Rating: {r['rating']}")
                else:
                    print("😕 No similar books found.")
            else:
                print(f"🔍 Processing query: {user_input}")
                recommendations = smart_book_recommend(user_input)
                if recommendations:
                    print("\n📘 Book Matches:")
                    for idx, r in enumerate(recommendations, 1):
                        print(f"{idx}. {r['title']} by {r['author']}")
                        if 'genre' in r:
                            print(f"   Genre: {r['genre']}")
                        if 'rating' in r:
                            print(f"   Rating: {r['rating']}")
                        if 'price' in r:
                            print(f"   Price: ${r['price']}")
                else:
                    print("😕 No books matched your query. Try again with a different input.")