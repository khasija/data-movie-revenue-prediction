import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasicPreprocessing(TransformerMixin, BaseEstimator):
    def __init__(self, train_set=True):
        self.train_set = train_set

    # def fit(self, X, y=None):
    #     return self

    def apply(self, X, y=None):
        """
        Function to return the clean dataframe after applying basic preprocessing steps
        Args: data = dataframe
        Returns: data_cleaned =  dataframe
        """
        # make copy of data
        df = X.copy()
        if self.train_set:
            # 1. keep only dataset with revenue greater than 3K and budget greater than 30k
            df = df[df["revenue"] > 3000]
            df = df[df["revenue"] < 1519557910]

            df = df[df["budget"] > 30000]
            df = df[df["budget"] < 260000000]
        # else:
        #     df = df[(df['budget'] > 300000) & (df['budget'] < 260000000)]

        # 2. Remove Duplicates
        df.drop_duplicates(inplace=True)

        # 3.add column with 1 if movie belongs to any collection and 0 if it does not belong to any collection
        df["collection"] = df["belongs_to_collection"].apply(lambda x: 0 if type(x) == float and pd.isna(x) else 1)

        # 4. Deal with missing Values
        ## 4.1 Drop columns
        df.drop(
            columns=["imdb_id", "status", "belongs_to_collection", "spoken_languages"], inplace=True
        )  # popularity,'tagline', 'overview',

        ## 4.2 Drop all the rows where we release_date is misssing
        # drop rows with null values in numeric variables
        df.dropna(axis=0, how="any", subset=["release_date"], inplace=True)

        # geners
        most_common_genres = df["genres"].describe()["top"]
        df["genres"] = df["genres"].apply(
            lambda x: most_common_genres if x == {} else x
        )  # TODO: Should we say that there is no genre instead?

        # one-hot encode genres
        one_hot = df["genres"].str.get_dummies(sep="|")

        df = df.join(one_hot)

        return df
