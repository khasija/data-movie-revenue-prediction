import streamlit as st
import pandas as pd
import imdb
import shap
import matplotlib.pyplot as plt
import json
import requests
st.set_option('deprecation.showPyplotGlobalUse', False)
import pickle
from python_files.dataframe_transformer import DataframeTransformer
from python_files.genre_transformer import GenreTranformer
from python_files.cast_transformer import CastTransformer
import numpy as np

MODEL_PATH = "model/xgb_model.pkl"



import base64

@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/jpeg;base64,{encoded}">'
    return tag

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style

image_path = 'images/cinema-1269996.jpg'

st.write(background_image_style(image_path), unsafe_allow_html=True)

original_title = '<p style="font-family:Courier; color:White; font-size: 30px;">This app predicts the Movie Revenue!</p>'
st.markdown(original_title, unsafe_allow_html=True)

# Sidebar
# Header of Specify Input Parameters
# st.sidebar.header('Specify Input Movie Name')

def user_input_features(movie_name):
    try:
        ia = imdb.Cinemagoer()
        search = ia.search_movie_advanced(movie_name)

        # getting the id
        imdb_id = search[0].movieID
        url = f'https://api.themoviedb.org/3/find/tt{imdb_id}?api_key={st.secrets["tmdb_key"]}&external_source=imdb_id'
        response = requests.get(url)
        info = response.json()
        movie_id = info['movie_results'][0]['id']

        df = {}
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={st.secrets["tmdb_key"]}'
        response = requests.get(url)

        backdrop_path = response.json()['backdrop_path']
        st.image(
                "https://image.tmdb.org/t/p/original/" + backdrop_path,
                width=400, # Manually Adjust the width of the image as per requirement
            )
        df['budget'] = response.json()['budget']
        df['release_date'] = response.json()['release_date']
        df['production_companies'] = response.json()['production_companies'][0]['name']
        df['production_companies_number'] = len(response.json()['production_companies'])
        df['production_countries_number'] = len(response.json()['production_countries'])
        df['runtime'] = response.json()['runtime']
        df['production_countries'] = response.json()['production_countries'][0]['name']
        genres = response.json()['genres']
        genre_list = []
        for i in range(len(genres)):
            genre_list.append(genres[i]['name'])
        genre_list = "|".join(genre_list)
        df['genres'] = genre_list
        df['popularity'] = response.json()['popularity']
        df['vote_average'] = float(response.json()['vote_average'])
        df['vote_count'] = int(response.json()['vote_count'])
        df['belongs_to_collection'] = response.json()['belongs_to_collection']


        url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key=279ec8b5e677bfd655c30c6403e14469'

                # get the response
        response = requests.get(url)
        director_names = []
        producer_names = []
        director_number = 0
        producer_number = 0
        actor_number = 0
        for i in range(len(response.json()['cast'])):
            if response.json()['cast'][i]['known_for_department'].lower() == 'acting':
                actor_number+=1
        for i in range(len(response.json()['crew'])):
            if response.json()['crew'][i]['job'].lower() == 'director':
                director_number+=1
                director_names.append(response.json()['crew'][i]['name'])
            if response.json()['crew'][i]['job'].lower() == 'producer':
                producer_number+=1
                producer_names.append(response.json()['crew'][i]['name'])
        df['director_name'] = director_names[0]
        df['producer_name'] = producer_names[0]
        df['director_number'] = director_number
        df['producer_number'] = producer_number
        df['actor_number'] = actor_number
        features = pd.DataFrame(df, index=['Value'])

        return features
    except:
        st.write('Please write correct movie name')
        return pd.DataFrame()

sel_col, disp_col = st.columns(2)
movie_name = sel_col.text_input('Which movie do you want to predict?')
if st.button('Get Revenue'):
    if movie_name:
        df = user_input_features(movie_name)
        if not df.empty:
            st.header('Specified Input parameters')
            st.write(df)
            st.write('---')
            my_pipeline = pickle.load(open(MODEL_PATH,"rb"))

            prediction = my_pipeline.predict(df)

            st.header('Prediction of Revenue')

            prediction = np.expm1(prediction)
            st.write(prediction)
# st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
