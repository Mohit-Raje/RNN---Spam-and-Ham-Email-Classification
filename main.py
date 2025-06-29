import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , SimpleRNN , Dense
from tensorflow.keras.callbacks import EarlyStopping , TensorBoard
from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import datetime
import pickle
import streamlit as st

model=load_model('model.h5')

with open('tokenizer.pkl' , 'rb') as file:
    tokenizer=pickle.load(file)

ps=PorterStemmer()

def preprocess_text(text):
    review=re.sub('[^a-zA-Z]' , ' ' , text)
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    return review
    
def encode_and_pad_text(text):
    sequences=tokenizer.texts_to_sequences([text])
    padded_text=pad_sequences(sequences , maxlen=500 , padding='pre')
    return padded_text

def prediction(text):
    preprocessed_text=preprocess_text(text)
    encoded_text=encode_and_pad_text(preprocessed_text)
    prediction=model.predict(encoded_text)
    
    if prediction[0][0] < 0.5:
        target='ham'
    else:
        target='spam'
    return target , prediction[0][0]


st.title("SmartMail Classifier")
st.write("Enter the mail content")

user_input=st.text_area('Email')


if st.button('Classify'):
    target , proba = prediction(user_input)
    
    st.write(f'This is a {target} email')
    st.write(f'Probability : {proba}')
    
else:
    st.write('Please write the input')    
    