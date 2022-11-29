import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from python_files.fetch_missing_data import fetch_missing_data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

class Preprocessing:
    # count the number of possible returns per variable
    def total_count(data, header, topk=30):
        '''
        Function to return the topk results and the number of their occurence in the data set
        Args: data = dataframe
            header = string; column header
            topk = int; default is 30; amount of top results to be displayed
        Returns: total =  list of dictionaries
        '''
        
        total = data[header][data[header] != 'none'].value_counts().sort_values(ascending = False)[:topk]
        return list(total.index)
    
    def add_top_30(dataset, col, topk):
        '''
        Function to add top 30 results from column headers as separate columns to 
        dataframe
        Args: dataset = dataframe 
            col = string; column name
            topk = list; top k values in column
        Returns: dataset = dataframe
        '''
        counter = 0
        for item in topk:
            header_name = str(item[0])+'_name'
            dataset[header_name] = dataset[col].apply(lambda x: 1 if item[0] in x else 0)

        return dataset
    
    def clean(df, headers, list_top_30=[], train_set=True):
        '''
        Function to clean a dataframe
        Args: df = dataframe
            headers = list containing names of column headers that need 
                        to be converted from strings to dicts
            list_top_30 = list; default value is empty list; otherwise it can hold
                            list with lists of top 30 results from specific columns
            train = boolean; True by default; designates whether the dataframe is
                    a test or train dataset
        Returns: cleaned_df = cleaned dataframe
                total_top_k_var = list of top 30 results from particular columns
        '''
        df_copy = df.copy()
        
        # keep only dataset with revenue greater than 3K and budget greater than 30k
        df_copy = df_copy[(df_copy['revenue'] > 3000) & (df_copy['budget'] > 300000)]
        
        # remove duplicates
        df.drop_duplicates(inplace = True)
        
        ## Numerical Data Preprocessing        
        
        # convert the data type of Popularity
        df_copy['popularity'] = pd.to_numeric(df_copy['popularity'])
        
        # add budget-year-ratio
        df_copy['release_year'] = df_copy['release_year'].astype('int32')
        df_copy['budget_year_ratio'] = round(df_copy['budget']/df_copy['release_year'], 2)
        
        ## Categorical Data Preprocessing
        
        # add column with 1 if movie belongs to any collection and 0 if it does not belong to any collection
        df_copy['collection'] = df_copy['belongs_to_collection'].apply(lambda x: 1 if x != 'missing_value' else 0)
        
        # add year
        df_copy['release_year'] = df_copy['release_date'].map(lambda x: str(19) + x[-2:] if int(x[-2:]) > 17 else str(20) + x[-2:])
        df_copy['release_year'] = df_copy['release_year'].astype('int32')
        # add month
        df_copy['release_month'] = df_copy['release_date'].map(lambda x: int(x[:2]) if x[1] != '/' else int(x[:1]))
        # add week
        df_copy['release_week'] = df_copy['release_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').isocalendar()[1])
        # add weekday
        df_copy['release_weekday'] = df_copy['release_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').strftime('%A'))
        
        # one-hot encode month variables
        one_hot_month = pd.get_dummies(df['release_month'], prefix='month')
        # one-hot-encode weekday variable
        one_hot_weekday = pd.get_dummies(df['release_weekday'], prefix='weekday')
        
        df = df.join(one_hot_month)
        df = df.join(one_hot_weekday)
        
        # add top thirty values as columns for below features
        top_30_vars = ['actor1_name', 'director_name', 'producer_name', 'production_companies']
        if train_set:
            for var in top_30_vars:
                top_k_var = total_count(df, var)
                list_top_30.append(top_k_var)
                cleaned_df = add_top_30(df, var, top_k_var)
        else:
            for i in range(len(top_30_vars)):
                cleaned_df = add_top_30(df, top_30_vars[i], list_top_30[i])
                
        col_list = ['belongs_to_collection', 'genres', 'spoken_languages', 
                    'production_companies', 'production_countries', 
                    'original_language', 'original_title', 'status', 'poster_path', 'release_date', 'release_month',
                    'release_weekday', 'id', 'imdb_id', 'overview', 'tagline']
    
        for item in col_list:
            cleaned_df.drop(item, axis=1, inplace=True)
        
        

