import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GenreTranformer(TransformerMixin, BaseEstimator):
    # BaseEstimator generates the get_params() and set_params() methods that all Pipelines require
    # TransformerMixin creates the fit_transform() method from fit() and transform()

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.unique_genre_list = []
        for genres in X:
            for genre in genres[0].split("|"):
                if genre not in self.unique_genre_list:
                    self.unique_genre_list.append(genre)
        #         print(len(self.unique_genre_list))
        return self

    def transform(self, X, y=None):
        unique_genre_list = self.unique_genre_list
        genre_transformed = []
        for genres in X:
            genre_np = np.zeros((len(unique_genre_list),), dtype=int)
            for target_genre in genres[0].split("|"):
                if target_genre in unique_genre_list:
                    index = unique_genre_list.index(target_genre)
                    genre_np[index] = 1
            genre_transformed.append(list(genre_np))
        return np.array(genre_transformed)
