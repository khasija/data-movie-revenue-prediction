import streamlit as st
import pandas as pd
import numpy as np
import pickle

from python_files.data import GetData

# load the model from disk
loaded_model = pickle.load(open('xiaohu_predict_model.sav', 'rb'))

#data = GetData().get_data()['AllDataMerged_updated']

# Score the model
#lin_model.score(X_test_trainsformed,y_test)

# Load data
data = GetData().get_data()['AllDataMerged_updated']

# Predict model
X_pred = pd.read_csv('../data/X_pred.csv')
y_pred_log = loaded_model.predict(X_pred)

#@st.cache
def get_dataframe_data():

    return pd.DataFrame(
            pd.DataFrame(X_pred)['budget_log'],
            pd.DataFrame(X_pred)['runtime'],
            y_pred_log
            
        )

df = get_dataframe_data()

st.write(df.head(10))

st.header(“Movie Revenue Prediction”)
col1 = st.columns(1)
with col1:
st.text(“X_pred”)
budget = st.slider(‘Budget’, pd.DataFrame(X_pred)['budget_log'])
runtime = st.slider(‘Runtime’, pd.DataFrame(X_pred)['runtime'])
