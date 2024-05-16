import pandas as pd
from song_index import SongIndex
from search_engine import RandomHyperplaneSearchEngine


def load_songs(filepath):
    """Load songs from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset="id")
    df = df.dropna()
    return df


def initialize_system(songs):
    """Initialize the song index and search engine with the given songs."""
    song_index = SongIndex(songs)
    return song_index


def search_songs(song_index, song_id):
    """Search for songs similar to the given song ID."""
    indices = song_index.search(song_id, k=10)
    return [
        (song_index.songs[idx]["name"], song_index.songs[idx]["artists"])
        for idx in indices
    ]


def main():
    """Main function to run the application."""
    filepath = "./data/tracks_features.csv"
    songs_df = load_songs(filepath)

    # Initialize the song index and search engine system
    song_index = initialize_system(songs_df)

    print("Welcome to the Music Recommendation System!")
    print("To begin, enter a song title and artist (e.g. 'Reptilia by The Strokes')")
    while True:
        query = input(">")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        similar_songs = song_index.search_by_name(query, k=10)
        for i in similar_songs:
            print(f"{song_index.songs[i]['name']} by {song_index.songs[i]['artists']}")


if __name__ == "__main__":
    main()
