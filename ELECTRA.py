import streamlit as st
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="google/electra-small-generator",
    tokenizer="google/electra-small-generator"
)

mask = fill_mask.tokenizer.mask_token

def init():
  st.write('# Masked Language Modeling')

  selected = st.selectbox('Phase', (
    f'How old {mask} you. ?',
    f'The {mask} of the United States is Joe Biden.', 
    f'Moscow is the {mask} of Russia.',
    f'Once upon a {mask} , two people lived.',
  ))

  result = fill_mask(selected)
  st.write(result)

if __name__ == '__main__':
  init()