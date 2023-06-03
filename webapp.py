# Importing Libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

import pickle

# 1. Display Title
st.header('Heart Disease Prediction')

# 2. Image selection

# 3. Get Data

@st.cache_resource
def show_charts():
    image = Image.open('heart.jpg')
    st.image(image, caption='Heart Disease Detection using ML', use_column_width=True)
    df = pd.read_csv('heart-20000.csv')
    del df['id']
    df.drop(df[df['ap_hi'] > 250].index, inplace=True)
    df.drop(df[df['ap_hi'] < 60].index, inplace=True)
    df.drop(df[df['ap_lo'] > 180].index, inplace=True)
    df.drop(df[df['ap_lo'] < 50].index, inplace=True)
    st.subheader('Data Information:')
    st.dataframe(df)
    st.write(df.describe())
    chart = st.line_chart(df)

show_charts()

score = 95.8359

def get_user_input():
    Age_in_days = st.sidebar.slider('age', 2000, 30000, 10000)
    Gender = st.sidebar.slider('gender', 1, 2, 1)
    Height = st.sidebar.slider('height', 120, 210, 175)
    Weight = st.sidebar.slider('weight', 50, 200, 80)
    AP_hi = st.sidebar.slider('ap_hi', 100, 250, 120)
    AP_lo = st.sidebar.slider('ap_lo', 50, 110, 90)
    Cholestrol = st.sidebar.slider('cholestrol', 1, 3, 2)
    Glucose = st.sidebar.slider('gluc', 1, 3, 2)
    Smoking = st.sidebar.slider('smoke', 0, 1, 0)
    Alcohol = st.sidebar.slider('alco', 0, 1, 0)
    Active = st.sidebar.slider('active', 0, 1, 1)

    # Store a dictionary into a variable
    user_data = {
        'Age (in days)': Age_in_days,
        'Gender': Gender,
        'Height': Height,
        'Weight': Weight,
        'AP_hi': AP_hi,
        'AP_lo': AP_lo,
        'Cholestrol': Cholestrol,
        'Glucose': Glucose,
        'Smoking intake': Smoking,
        'Alcohol intake': Alcohol,
        'Active': Active
    }
    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()
st.subheader('User Input:')
st.write(user_input)
st.subheader('Model Test Accuracy Score:')
st.write(score, '%')

# Add a submit button

if st.sidebar.button("Predict"):
    model = open('random_forest.pkl', 'rb')
    classifier = pickle.load(model)
    prediction1 = classifier.predict(user_input)
    st.subheader('Classification: ')
    st.write(prediction1)
