import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration


def summarize_text(url,model, tokenizer):
    # facebook/bart-large-xsum
    
    # https://arxiv.org/abs/1910.13461
    url = url
    response = requests.get(url)
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.text
    
    max_length = 1024
    truncated_input = text[:max_length]

    
    # Transform input tokens 
    inputs = tokenizer.encode(truncated_input, return_tensors="pt")

    # Generate summary
    summary_ids = model.generate(inputs, num_beams=4, max_length=1024, early_stopping=True)

    # Convert summary IDs to text
    summary_text = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary_text

#add cathc load model
@st.cache
def load_model():
    model_name = "facebook/bart-large-xsum" 
    # Download pytorch model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model,tokenizer

# Streamlit app
def main():
    model,tokenizer = load_model()

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