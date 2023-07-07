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
from transformers import AutoModel, AutoTokenizer 
import torch

def summarize_text(url,model, tokenizer):
    # facebook/bart-large-xsum
    
    # https://arxiv.org/abs/1910.13461
    url = url
    response = requests.get(url)
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.text
    
    


    # Transform input tokens 
    inputs = tokenizer(text=text, return_tensors="pt")

    # Model apply
    outputs = model(**inputs)

    return outputs

# Streamlit app
def main():
    # Define the model repo
    model_name = "facebook/bart-large-xsum" 
    # Download pytorch model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    st.title("Webpage Text Summarizer")
    st.write("Enter a URL and get a summary of the webpage text.")
    # Input URL
    url = st.text_input("Enter URL:")
    if url:
        try:
            summary = summarize_text(url,model,tokenizer)
            st.subheader("Summary:")
            st.write(summary)
        except:
            st.write("Error: Failed to summarize the text. Please check the URL or try again later.")

if __name__ == "__main__":
    main()