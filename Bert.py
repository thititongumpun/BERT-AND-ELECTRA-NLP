import streamlit as st
from transformers import pipeline

mlm = pipeline('fill-mask')
mask = mlm.tokenizer.mask_token

st.write('# Masked Language Modeling')

selected = st.selectbox('Phase', (
  f'Hello. How are {mask}', 
  f'At a {mask} you can drink beer and wine',
  f'Read the rest of this {mask} to understand things in more detail',
  f'I like to play {mask} with my friend',
  f'I {mask} going to work',
  f'Performing additions and subtractions is a part of {mask}'
))

result = mlm(selected)
st.write(result)
