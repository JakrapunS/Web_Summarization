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
    #transformed_text = re.sub(r'[^ ]', r'\\', text)
    max_length = 1024

    text_len = len(text)

    n = round(text_len/max_length)
    # Create text chunk
    chunk=[]
    for i in range(n):
        start = (i*max_length)
        end = (i+1)*max_length
        if end > text_len:
            end = text_len
            
        chunk.append(text[start:end])

    #print(truncated_input)
    # Transform input tokens 

    result_text = []
    for text in chunk:
        inputs = tokenizer.encode(text, return_tensors="pt")

        # Generate summary
        summary_ids = model.generate(inputs, num_beams=4, max_length=1024, early_stopping=True)

        # Convert summary IDs to text
        summary_text = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

        result_text.append(summary_text)

    concatenated_text = ' '.join(result_text)

    if len(concatenated_text)  > max_length:
        final = concatenated_text[:max_length]
    else:
        final =concatenated_text
    
    #final summarization
    #inputs_final = tokenizer.encode(final, return_tensors="pt")

    # Generate summary
    #summary_ids_final = model.generate(inputs_final, num_beams=4, max_length=1024, early_stopping=True)

    # Convert summary IDs to text
    #summary_text_final = tokenizer.decode(summary_ids_final.squeeze(), skip_special_tokens=True)

    return concatenated_text

#add cathc load model

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

            # Allow user to ask questions
            st.subheader("Question Answering:")
            question = st.text_input("Ask a question:")
            if question:
                try:
                    answer = answer_question(qa_pipeline, question, text)
                    st.write("Answer:", answer)
                except:
                    st.write("Error: Failed to answer the question.")
        except:
            st.write("Error: Failed to summarize the text. Please check the URL or try again later.")

if __name__ == "__main__":
    main()