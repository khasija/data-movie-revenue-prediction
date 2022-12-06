import csv
import os

import pandas as pd


class GetData:
    def __init__(self) -> None:
        self.data = None

    def get_data(self):
        """
        This function returns a dataframe.
        Its values should be pandas.DataFrames loaded from csv files
        """

        # Hints 1: Build csv_path as "absolute path" in order to call this method from anywhere.
        # Do not hardcode your path as it only works on your machine ('Users/username/code...')
        # Use __file__ instead as an absolute path anchor independant of your usename
        # Make extensive use of `breakpoint()` to investigate what `__file__` variable is really
        # Hint 2: Use os.path library to construct path independent of Mac vs. Unix vs. Windows specificities
        # YOUR CODE HERE
        if self.data == None:
            temp_path = os.path.dirname(__file__)
            path = temp_path[: -len("/python_files")]
            csv_path = path + "/data"

            file_names = []
            for item in os.listdir("../data/"):
                if item.endswith(".csv"):
                    file_names.append(item)

            key_names = []
            for file_name in file_names:
                if file_name.endswith(".csv"):
                    file_name = file_name[: -len(".csv")]
                    # if(file_name.endswith('.csv')):
                    #     file_name = file_name[:-4]
                    # if(file_name.startswith('olist')):
                    #     file_name = file_name.replace('olist_', '')
                    key_names.append(file_name)

            data_temp = {}
            for (key, value) in zip(key_names, file_names):
                if key == "AllMoviesCastingRaw":
                    data_temp[key] = pd.read_csv(
                        f"../data/{value}", delimiter=";", low_memory=False, encoding="ISO-8859-1"
                    )
                else:
                    data_temp[key] = pd.read_csv(f"../data/{value}", low_memory=False, encoding="ISO-8859-1")

            self.data = data_temp

        return self.data

    def get_merge_data():

        df = GetData().get_data()["AllMoviesDetailsCleaned"]

        # fetch the cast details as dataframe and merge it to df_copy
        df_cast = GetData().get_data()["AllMoviesCastingRaw"]
        df = df.merge(df_cast, on="id", how="left")

        return df
