import numpy as np
from search_engine import RandomHyperplaneSearchEngine
from utils.data_utils import normalize_features, compute_stats
from thefuzz import process


class SongIndex:
    def __init__(self, songs_df):
        self.songs = songs_df.to_dict(orient="records")
        self.feature_vectors = []
        self.song_info = {}
        self.song_id_to_index = {}
        self.mean = None
        self.std = None
        self._precompute_stats()
        self._vectorize_songs()
        self.search_engine = RandomHyperplaneSearchEngine(num_planes=10)
        self.search_engine.prepare_engine(self.feature_vectors)

    def _precompute_stats(self):
        numeric_features = np.array(
            [
                [
                    song["danceability"],
                    song["energy"],
                    song["loudness"],
                    song["speechiness"],
                    song["acousticness"],
                    song["instrumentalness"],
                    song["liveness"],
                    song["valence"],
                    song["tempo"],
                    song["duration_ms"],
                ]
                for song in self.songs
            ]
        )
        self.mean, self.std = compute_stats(numeric_features)

    def _encode_key(self, key):
        one_hot = np.zeros(12)
        one_hot[key] = 1
        return one_hot

    def _vectorize_songs(self):
        for song in self.songs:
            self.song_id_to_index[song["id"]] = len(self.feature_vectors)
            numeric_features = np.array(
                [
                    song["danceability"],
                    song["energy"],
                    song["loudness"],
                    song["speechiness"],
                    song["acousticness"],
                    song["instrumentalness"],
                    song["liveness"],
                    song["valence"],
                    song["tempo"],
                    song["duration_ms"],
                ]
            )
            normalized_features = normalize_features(
                numeric_features, self.mean, self.std
            )
            key_features = self._encode_key(song["key"])
            feature_vector = np.concatenate([normalized_features, key_features])
            self.feature_vectors.append(feature_vector)
            key = f"{song['name'].lower()} by {song['artists'].lower()}"
            self.song_info[key] = song["id"]

    def search_by_name(self, query, k=10):
        query = query.lower()
        # Find the best match using fuzzy matching
        best_match = process.extractOne(query, self.song_info.keys())
        if best_match and best_match[1] > 80:  # Threshold for match quality, e.g., 80%
            song_id = self.song_info[best_match[0]]
            return self.search(song_id, k)
        return "Song not found"

    def search(self, song_id, k=10):
        song_index = self.song_id_to_index.get(song_id, None)
        if song_index is None:
            return "Song not found"
        song_vector = self.feature_vectors[song_index]
        indices = self.search_engine.query(self.feature_vectors, song_vector, top_n=k)
        return indices
