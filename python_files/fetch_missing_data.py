import json

import requests

# function to fetch missing data


def fetch_missing_data(df):
    """This function tries to find missing data by hitting an api to fetch tmdb data"""

    # iterate through rows in df
    for row in df.itertuples():
        # hit the url
        url = f"https://api.themoviedb.org/3/movie/{row.id}?api_key=279ec8b5e677bfd655c30c6403e14469"

        # get the response
        response = requests.get(url)

        try:
            df.loc[row.id, ("budget_new")] = int(json.dumps(response.json()["budget"]))
            # df.loc[row.id, ('revenue_new')] = int(json.dumps(response.json()['revenue']))
            df.loc[row.id, ("popularity_new")] = json.dumps(response.json()["popularity"])
            df.loc[row.id, ("vote_average_new")] = float(json.dumps(response.json()["vote_average"]))
            df.loc[row.id, ("vote_count_new")] = int(json.dumps(response.json()["vote_count"]))
            df.loc[row.id, ("belongs_to_collection")] = json.dumps(response.json()["belongs_to_collection"])
        except:
            continue
    return df
