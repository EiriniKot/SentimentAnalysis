import streamlit as st
import os
import re
import sys
ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)
# sys.path.append(ROOT_DIR + '/src/utils')
from src.utils import preprocessing_tools as pt

import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import TweetTokenizer

import seaborn as sns
import tensorflow as tf
from keras.models import load_model

from PIL import Image

last = ROOT_DIR.split('/')[-1]
if last == 'SentimentAnalysis':
    pass
else:
    ROOT_DIR = ROOT_DIR+'/SentimentAnalysis'

image_path = ROOT_DIR + '/change-mood-positive-and-good-vs-negative-vector-36741242.jpg'
image = Image.open(image_path)
#displaying the image on streamlit app
st.image(image, caption=None)

col1, col2, col3 = st.columns(3)

with col1:
    st.title('Tweets Classifier')

with col2:
    st.image(image)

with col3:
    st.write(' ')

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.write(' ')

with col3:
    st.image(image)


sns.set_theme(style="darkgrid")
sns.set()
input_sentence = st.text_input(label='Feed me a tweet')

tk = TweetTokenizer()
stemmer = SnowballStemmer('english').stem
lemmatizer = WordNetLemmatizer().lemmatize

path = ROOT_DIR+'/models/'
filenames = os.listdir(path)
mdls = {}

for filename in filenames:
  model_dir = os.path.join(path, filename)
  model = load_model(model_dir)
  mdls[filename] = model

def custom_standardize(sentence):
  """
  This function is responsible to apply some preprocessing
  techniques to the sentences.
    1. First, it removes urls
    2. It removes tags
    3. It converts emoticons to their description
       sentence (for example :) to happy_face)
    4. Splits sentences using tweet tokenizer
    5. Removes stopwords
    6. Applies stemming and lemmatization
    7. Returns merged sentence

  """
  sentence = str(sentence)
  # Remove Urls
  sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',sentence)
  # Remove Tags
  sentence = re.sub('@[^\s]+','', sentence)
  # Replace emoticons with description
  sentence = pt.convert_emoticons(sentence)
  sentence = pt.custom_split(sentence, tk =tk)
  # Remove Stopwords
  sentence = list(set(sentence)-STOPWORDS)
  sentence = list(map(lambda i: pt.text_normalization(i,
                                                      stemmer = stemmer,
                                                      lemmatizer = lemmatizer),
                      sentence))
  # join elements of text with space
  sentence = ' '.join(sentence)
  print('-----',sentence)
  return sentence

if st.button('Give me a prediction'):
    sentence = custom_standardize(input_sentence)
    X_test = tf.convert_to_tensor(sentence)
    X_test = mdls['model_text_vec.tf'](X_test)
    model = mdls['model_0.h5']
    y_pred = model.predict(tf.expand_dims(X_test, 0))
    y_pred = tf.squeeze(y_pred)

    index = tf.argmax(y_pred)
    if index == 2:
        label = 'Positive'
    elif index == 1:
        label = 'Neutral'
    else:
        label = 'Negative'

    st.text_area(label="Output:", value=label)
