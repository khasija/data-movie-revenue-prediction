
class BasicPreprocessing:
    def __init__(self) -> None:
        pass
    
    def apply(data):
        '''
        Function to return the clean dataframe after applying basic preprocessing steps
        Args: data = dataframe            
        Returns: data_cleaned =  dataframe
    '''
        # make copy of data
        df = data.copy()
        
        # 1. keep only dataset with revenue greater than 3K and budget greater than 30k
        df = df[(df['revenue'] > 3000) & (df['budget'] > 300000)]
    
        # 2. Remove Duplicates
        df.drop_duplicates(inplace = True)
        
        # 3.add column with 1 if movie belongs to any collection and 0 if it does not belong to any collection
        df['collection'] = df['belongs_to_collection'].apply(lambda x: 1 if x != 'missing_value' else 0)
        
        # 4. Deal with missing Values
        ## 4.1 Drop columns
        df.drop(columns = ['popularity', 'tagline', 'overview', 'imdb_id', 'status', 'belongs_to_collection', 'spoken_languages'], inplace = True)
        
        ## 4.2 Drop all the rows where we release_date is misssing
        # drop rows with null values in numeric variables
        df.dropna( axis=0, how='any', subset=['release_date'], inplace= True )
        
          
        
        return df
        
        