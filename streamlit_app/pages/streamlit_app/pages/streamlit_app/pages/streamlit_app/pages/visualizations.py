import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title('Travel Data Visualizations')


# Load data
flights = pd.read_csv('data/raw/flights.csv')


# Flight price distribution
st.subheader('Flight Price Distribution')

fig, ax = plt.subplots()

ax.hist(flights['price'])

st.pyplot(fig) 
