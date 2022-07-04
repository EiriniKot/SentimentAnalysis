import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()

st.title('Tweets Classifier')
st.text_input(label = 'input')

# def preprocess_me(input):
#   sample_ = input
#   return sample_


# sample = preprocess_me(input)
