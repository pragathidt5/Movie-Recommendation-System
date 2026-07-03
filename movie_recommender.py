import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie data
movies = {
    "title": [
        "Avatar",
        "Titanic",
        "Avengers",
        "Iron Man",
        "Batman",
        "Superman",
        "Spider-Man",
        "Doctor Strange",
        "Black Panther",
        "Thor"
    ],
    "genre": [
        "Action Adventure Sci-Fi",
        "Romance Drama",
        "Action Superhero",
        "Action Superhero",
        "Action Crime",
        "Action Superhero",
        "Action Superhero",
        "Fantasy Action",
        "Action Adventure",
        "Fantasy Action"
    ]
}

# Create DataFrame
df = pd.DataFrame(movies)

# Convert genres into vectors
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df["genre"])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

print("=" * 50)
print("      MOVIE RECOMMENDATION SYSTEM")
print("=" * 50)

movie_name = input("\nEnter a movie name: ")

if movie_name in df["title"].values:
    index = df[df["title"] == movie_name].index[0]

    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:\n")

    count = 1
    for movie in similarity_scores[1:6]:
        print(f"{count}. {df.iloc[movie[0]].title}")
        count += 1

    print("\nThank you for using the Movie Recommendation System!")

else:
    print("\n❌ Sorry! Movie not found in the database.")