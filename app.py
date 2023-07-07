from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from transformers import BertModel, BertTokenizer
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re
from transformers import BertTokenizerFast, EncoderDecoderModel, AutoTokenizer
import torch

def summarize_text(url,sentense_fuser, tokenizer):
    
    url = url
    response = requests.get(url)
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.text
    sentence_fuser = sentense_fuser
    tokenizer = tokenizer

    input_ids = tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    outputs = sentence_fuser.generate(input_ids)

    return tokenizer.decode(outputs[0])
    

# Streamlit app
def main():
    st.title("Webpage Text Summarizer")

    st.write("Enter a URL and get a summary of the webpage text.")
    sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
    tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")
    # Input URL
    url = st.text_input("Enter URL:")
    if url:
        try:
            summary = summarize_text(url,sentence_fuser,tokenizer)
            st.subheader("Summary:")
            st.write(summary)
        except:
            st.write("Error: Failed to summarize the text. Please check the URL or try again later.")

if __name__ == "__main__":
    main()