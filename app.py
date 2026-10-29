import streamlit as st
import re
import pickle
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
words=stopwords.words('english')
with open('prediction_pipeline.pkl', 'rb') as f:
    model=pickle.load(f)

def cleaner(col):
  col=re.sub('<.*?>','',col)
  col=re.sub('https?://\S+|www\.\S+','',col)
  return re.sub('[^\w\s]','',col)
def stopword_removal(col):
  L=[]
  tokens=word_tokenize(col)
  for i in tokens:
    if i not in words:
      L.append(i)
  return " ".join(L)
sbs=SnowballStemmer('english')
def stemmer(col):
  L=[]
  tokens=word_tokenize(col)
  for i in tokens:
    L.append(sbs.stem(i))
  return " ".join(L)
def predict(text:str):
  text=cleaner(text)
  text=stopword_removal(text)
  text=stemmer(text)
  pred=model.predict([text])
  if pred[0]==0:
    return "Negative"
  return "Positive"

st.title('SENTIMENT ANALYSIS')
message=st.text_input('Enter the message: ')
if st.button('Tell Sentiment'):
  st.header('Your message :')
  st.subheader(message)
  response=predict(message)
  if response=='Negative':
    st.warning(f"The sentiment for message is {response}")
  else:
    st.success(f"The sentiment of message is {response}")
