from django.shortcuts import render,redirect
from django.http import JsonResponse
from .models import Movie
import sqlite3
from django.db import connection
import numpy as np
from tensorflow.keras.models import load_model
# Create your views here.
def index(request):
   return render(request,"login.html",{})




def movie_list(request):
    with connection.cursor() as cursor:
        # Fetch all movies with title, poster, description, and rating
        cursor.execute("SELECT Series_Title, Poster_Link, Overview, IMDB_Rating FROM movies")
        movies = cursor.fetchall()

    # Convert tuples to a list of dictionaries
    movie_data = [
        {'title': row[0], 'poster': row[1], 'description': row[2], 'rating': row[3]}
        for row in movies
    ]

    return render(request, 'list.html', {'movies': movie_data})

from django.shortcuts import render
from django.db import connection
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# Load the trained content-based model
content_nn = load_model("D:/ml_hackathon/movie_recommendation/content_based_model.keras")


def fetch_movie_data():
    """Retrieve movie data from the database."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT Series_Title, Poster_Link, Overview, IMDB_Rating FROM movies")
        movies = cursor.fetchall()

    df = pd.DataFrame(movies, columns=["Series_Title", "Poster_Link", "Overview", "IMDB_Rating"])
    return df


def recommend_movies_content_based(movie_title, df):
    """Recommend movies using the trained neural network model."""

    # Check if movie exists
    if movie_title.lower() not in df["Series_Title"].str.lower().values:
        return []

    # Find index of input movie
    movie_idx = df[df["Series_Title"].str.lower() == movie_title.lower()].index[0]

    # Use TF-IDF Vectorizer for text embedding
    vectorizer = TfidfVectorizer(stop_words="english")
    movie_overview_matrix = vectorizer.fit_transform(df["Overview"].fillna(""))

    # Compute similarity scores
    similarity_scores = cosine_similarity(movie_overview_matrix[movie_idx], movie_overview_matrix)

    # Get top 5 similar movies (excluding itself)
    top_n_indices = similarity_scores.argsort()[0][-6:-1][::-1]

    return df.iloc[top_n_indices].to_dict(orient="records")


def recommend_movies(request):
    """Handle movie recommendation logic."""
    df = fetch_movie_data()

    if request.method == "POST":
        movie_title = request.POST.get("movie_title")
        recommended_movies = recommend_movies_content_based(movie_title, df)

        return render(request, "recommendation.html", {"movies": recommended_movies})

    return render(request, "list.html", {"movies": df.to_dict(orient="records")})
from django.contrib.auth import authenticate, login
from django.contrib import messages
def login(request):
    return render(request,"login.html",{})
def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("list")  # Redirect to home page after login
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "login.html")

def signup_data(request):
    email = request.GET['email']
    password = request.GET['password']

    con = sqlite3.connect("db.sqlite3")
    cur = con.cursor()
    cur.execute("insert into users(email,password) values(?,?,?)",(email,password))
    con.commit()
    return redirect("/index")
def redirevt(request):
    return redirect("main")
def list_mov(request):
    df = fetch_movie_data()
    return render(request, "list.html", {"movies": df.to_dict(orient="records")})