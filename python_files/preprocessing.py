import pandas as pd
from python_files.data import GetData
from sklearn.model_selection import train_test_split
from python_files.fetch_missing_data import fetch_missing_data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from datetime import datetime


class Advancedprocessing:
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
    
    def process(df, list_top_30=[], train_set=True):
        '''
        Function to clean a dataframe
        Args: df = dataframe            
            list_top_30 = list; default value is empty list; otherwise it can hold
                            list with lists of top 30 results from specific columns
            train = boolean; True by default; designates whether the dataframe is
                    a test or train dataset
        Returns: cleaned_df = cleaned dataframe
                total_top_k_var = list of top 30 results from particular columns
        '''
        df_copy = df.copy()
        
        # fetch the cast details as dataframe and merge it to df_copy
        df_cast = GetData().get_data()['AllMoviesCastingRaw']
        df_copy = df_copy.merge(df_cast, on = 'id', how = 'left')         
        
        # drop rows with null values in numeric variables
        # df_copy = df_copy.dropna(axis=0, how='any', subset=['release_date'])
        df_copy['release_date'] = pd.to_datetime(df_copy['release_date'], infer_datetime_format=True)
        
        ## Numerical Data Preprocessing        
        
        # convert the data type of Popularity
        # df_copy['popularity'] = pd.to_numeric(df_copy['popularity'])
        
        # add year
        df_copy['release_year'] = df_copy['release_date'].dt.year
        # df_copy['release_year'] = df_copy['release_year'].astype('int32')
        # add month
        df_copy['release_month'] = df_copy['release_date'].dt.month
        # add week
        df_copy['release_week'] = df_copy['release_date'].dt.dayofweek
        # add weekday
        df_copy['release_weekday'] = df_copy['release_date'].dt.day_name()
        
        # add budget-year-ratio
        df_copy['release_year'] = df_copy['release_year'].astype('int32')
        df_copy['budget_year_ratio'] = round(df_copy['budget']/df_copy['release_year'], 2)
        
        ## Categorical Data Preprocessing      
               
        # one-hot encode month variables
        # one_hot_month = pd.get_dummies(df['release_month'], prefix='month')
        # one-hot-encode weekday variable
        # one_hot_weekday = pd.get_dummies(df['release_weekday'], prefix='weekday')
        
        # df = df.join(one_hot_month)
        # df = df.join(one_hot_weekday)
        
        # add top thirty values as columns for below features
        top_30_vars = ['actor1_name', 'director_name', 'producer_name', 'production_companies']
        if train_set:
            for var in top_30_vars:
                top_k_var = Advancedprocessing.total_count(df_copy, var)
                list_top_30.append(top_k_var)
                cleaned_df = Advancedprocessing.add_top_30(df_copy, var, top_k_var)
        else:
            for i in range(len(top_30_vars)):
                cleaned_df = Advancedprocessing.add_top_30(df_copy, top_30_vars[i], list_top_30[i])
                
        col_list = [ 'genres', 
                    'production_companies', 'production_countries', 
                    'original_language', 'original_title', 'release_date', 'release_month',
                    'release_weekday', 'id']
    
        for item in col_list:
            cleaned_df.drop(item, axis=1, inplace=True)
            
         # Reset the index so I will be able to match the revenue, title and budget to the rows later on
        cleaned_df = cleaned_df.reset_index()
        
        return cleaned_df
        
        

