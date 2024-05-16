import numpy as np


class RandomHyperplaneSearchEngine:
    def __init__(self, num_planes=10):
        self.num_planes = num_planes
        self.hyperplanes = None
        self.hash_tables = {}

    def prepare_engine(self, feature_vectors):
        self.hyperplanes = np.random.randn(self.num_planes, len(feature_vectors[0]))
        self._build_hash_tables(feature_vectors)

    def _hash(self, vector):
        hash_value = 0
        for plane in self.hyperplanes:
            hash_value <<= 1
            if np.dot(vector, plane) >= 0:
                hash_value |= 1
        return hash_value

    def _build_hash_tables(self, feature_vectors):
        for i, vector in enumerate(feature_vectors):
            hash_value = self._hash(vector)
            if hash_value in self.hash_tables:
                self.hash_tables[hash_value].append(i)
            else:
                self.hash_tables[hash_value] = [i]

    def query(self, feature_vectors, song_vector, top_n=10):
        hash_value = self._hash(song_vector)
        candidate_indices = self.hash_tables.get(hash_value, [])
        distances = [
            (index, np.linalg.norm(feature_vectors[index] - song_vector))
            for index in candidate_indices
        ]
        distances.sort(key=lambda x: x[1])
        return [index for index, _ in distances[1 : top_n + 1]]
