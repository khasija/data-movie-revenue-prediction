import json

import imdb
import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap
import streamlit as st
import altair as alt
import time
from datetime import datetime

from typing import Tuple, Union

st.set_page_config(
    page_title="Alchemy", page_icon=":)", layout="wide", initial_sidebar_state="expanded", menu_items=None
)

st.set_option("deprecation.showPyplotGlobalUse", False)
import pickle

import numpy as np

from python_files.cast_transformer import CastTransformer
from python_files.dataframe_transformer import DataframeTransformer
from python_files.genre_transformer import GenreTranformer

MODEL_PATH = "model/xgb_model.pkl"
import base64

import math
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def load_pipeline():
    return pickle.load(open(MODEL_PATH, "rb"))


def predict(pipeline, data):
    return pipeline.predict(data)


def init_state():
    if "pipeline" not in st.session_state:
        st.session_state["pipeline"] = load_pipeline()
# Navigation bar
st.markdown("""
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            """, unsafe_allow_html = True)

def card_component(image, title, fun_fact):
    return f"""
        <div class="card" style="width: 25rem; background-color: rgba(220,220,220,0.2)">
        <img class="card-img-top" src={image} alt="Card image cap">
            <div class="card-body">
                <h5 class="card-title-dark">{title}</h5>
                <p class="card-text-dark">{fun_fact}</p>                
            </div>
        </div>
        """
def user_input_features(movie_name: str) -> Union[Tuple[pd.DataFrame, str], Tuple[None, None]]:
    try:
        # raise ValueError("fake error")
        ia = imdb.Cinemagoer()
        search = ia.search_movie_advanced(movie_name)

        # getting the id
        imdb_id = search[0].movieID
        url = f'https://api.themoviedb.org/3/find/tt{imdb_id}?api_key={st.secrets["tmdb_key"]}&external_source=imdb_id'
        response = requests.get(url)
        info = response.json()
        movie_id = info["movie_results"][0]["id"]

        df = {}
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={st.secrets["tmdb_key"]}'
        response = requests.get(url)

        backdrop_path = response.json()["backdrop_path"]

        df["budget"] = response.json()["budget"]
        df["release_date"] = response.json()["release_date"]
        df["production_companies"] = response.json()["production_companies"][0]["name"]
        df["production_companies_number"] = len(response.json()["production_companies"])
        df["production_countries_number"] = len(response.json()["production_countries"])
        df["runtime"] = response.json()["runtime"]
        df["production_countries"] = response.json()["production_countries"][0]["name"]
        genres = response.json()["genres"]
        genre_list = []
        for i in range(len(genres)):
            genre_list.append(genres[i]["name"])
        genre_list = "|".join(genre_list)
        df["genres"] = genre_list
        df["popularity"] = response.json()["popularity"]
        df["vote_average"] = float(response.json()["vote_average"])
        df["vote_count"] = int(response.json()["vote_count"])
        df["belongs_to_collection"] = response.json()["belongs_to_collection"]
        df["Revenue"] = response.json()["revenue"]

        url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key=279ec8b5e677bfd655c30c6403e14469"

        # get the response
        response = requests.get(url)
        director_names = []
        producer_names = []
        actor_names = []
        director_number = 0
        producer_number = 0
        actor_number = 0
        for i in range(len(response.json()["cast"])):
            if response.json()["cast"][i]["known_for_department"].lower() == "acting":
                if i < 6:
                    actor_name = response.json()["cast"][i]["name"]
                    actor_names.append(actor_name)
                actor_number += 1
        for i in range(len(response.json()["crew"])):
            if response.json()["crew"][i]["job"].lower() == "director":
                director_number += 1
                director_names.append(response.json()["crew"][i]["name"])
            if response.json()["crew"][i]["job"].lower() == "producer":
                producer_number += 1
                producer_names.append(response.json()["crew"][i]["name"])
        df["director_name"] = director_names[0]
        df["producer_name"] = producer_names[0]
        df["director_number"] = director_number
        df["producer_number"] = producer_number
        df["actor_number"] = actor_number
        actor_names = ', '.join(actor_names)     
        features = pd.DataFrame(df, index=["Value"])

        return features, backdrop_path, actor_names
    except:
        return None, None, None


def page1():
    # add_bg_from_local('images/curtain_background.jpg') 
    tab1, tab2 = st.tabs(["Exitsing movies", "New movie"])
    with tab1:
        with st.container():
            st.markdown(
                """
                        <style>
                        .big-font {
                        font-size:130px !important;
                        } </style>
                        """,
                unsafe_allow_html=True,
            )
            st.markdown('<p class="big-font">Life is too short for ordinary apps.</p>', unsafe_allow_html=True)
            st.markdown(
                """
                        <style>
                        .big-font2 {
                        font-size:100px !important;
                        } </style>
                        """,
                unsafe_allow_html=True,
            )
            st.markdown('<p class="big-font2">#FastForwardYourFuture</p>', unsafe_allow_html=True)

            # You can call any Streamlit command, including custom components:
            # st.bar_chart(np.random.randn(50, 3))

        # @st.cache
        # def load_image(path):
        #     with open(path, 'rb') as f:
        #         data = f.read()
        #     encoded = base64.b64encode(data).decode()
        #     return encoded

        # def image_tag(path):
        #     encoded = load_image(path)
        #     tag = f'<img src="data:image/jpeg;base64,{encoded}">'
        #     return tag

        # def background_image_style(path):
        #     encoded = load_image(path)
        #     style = f'''
        #     <style>
        #     .stApp {{
        #         background-image: url("data:image/jpeg;base64,{encoded}");
        #         background-size: cover;
        #     }}
        #     </style>
        #     '''
        #     return style

        # image_path = 'images/cinema-1269996.jpg'

        # st.write(background_image_style(image_path), unsafe_allow_html=True)

        # original_title = (
        #     '<p style="font-family:Courier; color:Black; font-size: 30px;">This app predicts the Movie Revenue!</p>'
        # )
        # st.markdown(original_title, unsafe_allow_html=True)
        
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="small")

        df, path, cast = None, None, None
        prediction = None

        with col1:
            option = st.radio("Please enter a movie or select from the list",
                            ('Enter Movie Name', 'Select from list'))
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            if(option == 'Enter Movie Name'):
                movie_name = st.text_input("Which movie's revenue do you want to predict?")
            elif(option == 'Select from list'):
                movie_name = st.selectbox("Which movie's revenue do you want to predict?",
                                    ('Parasite', 'Once Upon a Time in Hollywood', 'Aladdin', 'Nomadland'))

            if movie_name:
                # st.write(movie_name)
                
                with st.spinner('Wait for it...'):
                    df, path, cast = user_input_features(movie_name)
                st.success('Done!')

                if df is None and path is None:
                    st.write(f"Cannot find your movie '{movie_name}', please write correct movie name")

            if st.button("Get Revenue"):
                if df is None and path is None:
                    st.write("Please provide a movie name")
                else:                    
                    output_results = {'Budget(US Dollar)': millify(df["budget"][0]),
                              'Release Date': df["release_date"][0]}
                    
                    st.write(output_results)

                    prediction = predict(st.session_state["pipeline"], df)   
        

        with col2:
            if movie_name:                                          
                if df is not None and path is not None:
                    st.write("Movie Poster")
                    st.markdown(card_component(image="https://image.tmdb.org/t/p/original/" + path, 
                                               title= "Movie Cast", 
                                               fun_fact= f"Actors : {cast}")
                                , unsafe_allow_html=True)
                    
                    # st.image(
                    #     "https://image.tmdb.org/t/p/original/" + path,
                    #     width=400  # Manually Adjust the width of the image as per requirement
                    # )

        with col3:

            if prediction is not None:
                st.write("Acutal vs. Predicted Revenue") 
                
                act_vs_pred = {"Actual Revenue":  millify(df["Revenue"][0]),
                       "Predicted Revenue": millify(np.expm1(prediction[0]))}
                
                st.write(act_vs_pred)
                

                fig = {"Revenue": df["Revenue"][0], "Prediction": np.expm1(prediction[0])}
                fig2 = pd.DataFrame(fig, index = [0]).T
                st.bar_chart(data=fig2)


if __name__ == "__main__":
    init_state()
    page1()
