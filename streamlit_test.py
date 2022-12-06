import json

import imdb
import matplotlib.pyplot as plt
import pandas as pd
import requests
import shap
import streamlit as st
import altair as alt

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


def load_pipeline():
    return pickle.load(open(MODEL_PATH, "rb"))


def predict(pipeline, data):
    return pipeline.predict(data)


if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = load_pipeline()


tab1, tab2 = st.tabs(["Exitsing movies", "New movie"])
with tab1:
    with st.container():
        st.markdown(
            """
                    <style>
                    .big-font {
                    font-size:150px !important;
                    } </style>
                    """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="big-font">Believe the RMSE ...</p>', unsafe_allow_html=True)
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

    def user_input_features(movie_name):
        try:
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
            director_number = 0
            producer_number = 0
            actor_number = 0
            for i in range(len(response.json()["cast"])):
                if response.json()["cast"][i]["known_for_department"].lower() == "acting":
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
            features = pd.DataFrame(df, index=["Value"])

            return features, backdrop_path
        except:
            st.write("Please write correct movie name")
            return pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Which movie's revenue do you want to predict?")
        movie_name = col1.text_input("")
        df, path = user_input_features(movie_name)
        col1 = st.button("Get Revenue")

        st.subheader("Budget(US Dollar):")
        st.subheader(df["budget"][0])
        st.subheader("Release Date:")
        st.subheader(df["release_date"][0])

    _prediction = 0
    _revenue = 0
    with col2:

        if col2:
            if movie_name:
                df, path = user_input_features(movie_name)
                if not df.empty:
                    # st.header("Specified Input parameters")
                    # st.write(df)
                    # st.write("---")

                    prediction = predict(st.session_state["pipeline"], df)
                    revenue = df["Revenue"]
                    _revenue = [revenue[0]]
                    # st.header("Prediction of Revenue")
                    prediction = np.expm1(prediction)
                    _prediction = [prediction[0]]
                    st.image(
                        "https://image.tmdb.org/t/p/original/" + path,
                        width=400,  # Manually Adjust the width of the image as per requirement
                    )
                    # st.write(prediction[0])
                    # st.write(revenue)
                    # fig = {"Revenue": [revenue[0]], "Prediction": [prediction[0]]}
                    # fig2=pd.DataFrame(fig).T
                    # st.bar_chart(data = fig2)
                    # st.write(fig2)

                    # st.subheader("Budget(US Dollar)")
                    # st.subheader(df["budget"][0])

                    # st.subheader("Release Date")
                    # st.subheader(df["release_date"][0])

    with col3:
        st.header("Prediction of Revenue")
        st.subheader(prediction[0])
        #    revenue = df["Revenue"]
        #    st.header("Prediction of Revenue")
        #    prediction = np.expm1(prediction)
        #    st.write(prediction[0])
        #    st.write(revenue)
        fig = {"Revenue": _revenue, "Prediction": _prediction}
        fig2 = pd.DataFrame(fig).T
        st.bar_chart(data=fig2)
    #  st.write(fig2)

    #    st.subheader("Budget(US Dollar)")
    #    st.subheader(df["budget"][0])

    #    st.subheader("Release Date")
    #    st.subheader(df["release_date"][0])
