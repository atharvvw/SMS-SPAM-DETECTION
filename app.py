import streamlit as st 
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
   # 1. preprocess
   transformed email = transform(input_sms)
   # 2. vectorize
   vector_input = tfidf.transform(['transformed email'])
   # 3. predict
   result = model.predict(vector_input)[0] 
   # 4. Display
   if result == 1:
    st.header("Spam")
   else:
       st.header("Not Spam")
