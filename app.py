import streamlit as st
import requests
from bs4 import BeautifulSoup
#from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline
import torch



def summarize_text(url,summarizer, params):
    # facebook/bart-large-xsum
    
    # https://arxiv.org/abs/1910.13461
    url = url
    response = requests.get(url)
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.text

    result = summarizer(text, **params)

    return result[0]['summary_text']


def load_model():
    summarizer = pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device=0 if torch.cuda.is_available() else -1,
    )
    params = {
        "max_length": 1024,
        "min_length": 8,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "repetition_penalty": 3.5,
        "length_penalty": 0.1,
        "encoder_no_repeat_ngram_size": 3,
        "num_beams": 4,
    } # parameters for text generation out of model
    return summarizer,params


# Streamlit app
def main():
    summarizer,params = load_model()

    st.title("Webpage Text Summarizer")
    st.write("Enter a URL and get a summary of the webpage text.")

    example_url = "https://www.abc.net.au/news/2023-07-11/crown-resorts-450-million-fine-money-laundering/102588950"
    st.markdown(f"For instance, you can copy the following example URL:<br>[`{example_url}`]({example_url})", unsafe_allow_html=True)
    # Input URL
    url = st.text_input("Enter URL:")
    if url:
        try:
            summary = summarize_text(url,summarizer,params)
            st.subheader("Summary:")
            st.write(summary)
        except:
            st.write("Error: Failed to summarize the text. Please check the URL or try again later.")

if __name__ == "__main__":
    main()