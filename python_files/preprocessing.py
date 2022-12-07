from datetime import datetime

import numpy as np
import pandas as pd

from python_files.data import GetData


class Advancedprocessing:

    # count the number of possible returns per variable
    def total_count(data, header, topk=30):
        """
        Function to return the topk results and the number of their occurence in the data set
        Args: data = dataframe
            header = string; column header
            topk = int; default is 30; amount of top results to be displayed
        Returns: total =  list of dictionaries
        """

        total = data[header][data[header] != "none"].value_counts().sort_values(ascending=False)[:topk]
        return list(total.index)

    def add_top_30(dataset, col, topk):
        """
        Function to add top 30 results from column headers as separate columns to
        dataframe
        Args: dataset = dataframe
            col = string; column name
            topk = list; top k values in column
        Returns: dataset = dataframe
        """
        counter = 0
        for item in topk:
            header_name = str(item) + "_name"
            dataset[header_name] = dataset[col].apply(lambda x: 1 if item in x else 0)

        return dataset

    def process(df, list_top_30=[], train_set=True):
        """
        Function to clean a dataframe
        Args: df = dataframe
            list_top_30 = list; default value is empty list; otherwise it can hold
                            list with lists of top 30 results from specific columns
            train = boolean; True by default; designates whether the dataframe is
                    a test or train dataset
        Returns: cleaned_df = cleaned dataframe
                total_top_k_var = list of top 30 results from particular columns
        """
        df_copy = df.copy()

        # fetch the cast details as dataframe and merge it to df_copy
        # df_cast = GetData().get_data()['AllMoviesCastingRaw']
        # df_copy = df_copy.merge(df_cast, on = 'id', how = 'left')

        # drop rows with null values in numeric variables
        df_copy = df_copy.dropna(axis=0, how="any", subset=["release_date"])
        df_copy["release_date"] = pd.to_datetime(df_copy["release_date"], infer_datetime_format=True)

        ## Numerical Data Preprocessing

        # dealing with missing values
        # df_copy['runtime'] = df_copy['runtime'].fillna(df_copy['runtime'].mean())

        # add year
        df_copy["release_year"] = df_copy["release_date"].dt.year
        # df_copy['release_year'] = df_copy['release_year'].astype('int32')
        # add month
        df_copy["release_month"] = df_copy["release_date"].dt.month

        # add week
        df_copy['release_week'] = df_copy['release_date'].dt.isocalendar().week
        df_copy['week_sin'] = np.sin(2 * np.pi * df_copy['release_week']/52.0)
        df_copy['week_cos'] = np.cos(2 * np.pi * df_copy['release_week']/52.0)
        
        df_copy[['week_sin','week_cos']] = df_copy[['week_sin','week_cos']].astype('float64')
        df_copy[['week_sin','week_cos']] = df_copy[['week_sin','week_cos']].astype('float64')
        
        # df_copy[['week_sin', 'week_cos']].astype('float64', inplace = True)

        # add weekday
        df_copy["release_weekday"] = df_copy["release_date"].dt.day_name()
        # add age
        now = pd.to_datetime("now")
        df_copy["release_age"] = (now - df_copy["release_date"]).astype("<m8[Y]")

        # add budget-year-ratio
        df_copy["release_year"] = df_copy["release_year"].astype("int32")
        df_copy["budget_year_ratio"] = round(df_copy["budget"] / df_copy["release_year"], 2)

        ## Categorical Data Preprocessing

        # one-hot encode month variables
        # one_hot_month = pd.get_dummies(df_copy['release_month'], prefix='month')
        # one-hot-encode weekday variable
        # one_hot_weekday = pd.get_dummies(df_copy['release_weekday'], prefix='weekday')

        # df_copy = df_copy.join(one_hot_month)
        # df_copy = df_copy.join(one_hot_weekday)

        # add top thirty values as columns for below features
        top_30_vars = [
            "director_name",
            "producer_name",
            "production_companies",
            "production_countries",
        ]  #'actor1_name',
        if train_set:
            for var in top_30_vars:
                top_k_var = Advancedprocessing.total_count(df_copy, var)
                list_top_30.append(top_k_var)
                cleaned_df = Advancedprocessing.add_top_30(df_copy, var, top_k_var)
        else:
            for i in range(len(top_30_vars)):
                # print(top_30_vars[i])
                # print(list_top_30[i])
                cleaned_df = Advancedprocessing.add_top_30(df_copy, top_30_vars[i], list_top_30[i])

        # scale the budget and runtime
        3
        ## convert budget into logscale

        # df_copy['budget'] = df_copy['budget'].apply(lambda x: np.log(x + 1))

        col_list = [
            "genres",
            "title",
            "actor1_name",
            "actor2_name",
            "actor3_name",
            "actor4_name",
            "director_gender",
            "actor5_name",
            "director_name",
            "producer_name",
            "screeplay_name",
            "actor1_gender",
            "actor2_gender",
            "actor3_gender",
            "actor4_gender",
            "actor5_gender",
            "editor_name",
            "production_companies",
            "production_countries",
            "release_month",
            "budget",
            "spoken_languages_number",
            "original_language",
            "release_date",
            "release_year",
            "release_week",
            "original_title",
            "overview",
            "tagline",
            "release_weekday",
            "id",
        ]  #

        for item in col_list:
            cleaned_df.drop(item, axis=1, inplace=True)

        # Reset the index so I will be able to match the revenue, title and budget to the rows later on
        cleaned_df = cleaned_df

        return cleaned_df, list_top_30
