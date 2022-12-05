import numpy as np
import datetime
import pandas as pd

class DataframeTransformer:
    # @staticmethod
    # def names(X):
    #     return list(X_train.columns)

    @staticmethod
    def _has_collection(x):
        if pd.isnull(x):
            return 0
        return 1

    @staticmethod
    def transformer(X):

        X['release_date'] = pd.to_datetime(X['release_date'],infer_datetime_format=True)
        X['belongs_to_collection_updated'] = X['belongs_to_collection'].apply(DataframeTransformer._has_collection)
        # add weekday
        X['release_weekday'] = X['release_date'].dt.day_name()
        #Log of budget
#         X['log_budget'] = np.log(X['budget'])
        #find age of movie
        now = pd.to_datetime('now')
        X['release_age'] = (now - X['release_date']).astype('<m8[Y]')

        X['week_sin'] = np.sin(2 * np.pi * X['release_date'].dt.isocalendar().week/52)

        X['week_cos'] = np.cos(2 * np.pi * X['release_date'].dt.isocalendar().week/52)

        #budget to year ratio
        X['budget_year_ratio'] = round(X['budget']/X['release_date'].dt.year, 2)
        return X
