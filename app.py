import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer,pipeline, AutoModelForSeq2SeqLM
import transformers
import torch
#from langchain import PromptTemplate,  LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain




def summarize_text(url,model):
    
    url = url
    response = requests.get(url)
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.text

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    
    #template = """
    #          Write a concise summary of the following text.
    #          ```{text}```
    #       """
    #prompt = PromptTemplate(template=template, input_variables=["text"])
    #llm_chain = LLMChain(prompt=prompt, llm=model)
    chain = load_summarize_chain(model, chain_type='map_reduce')
    return chain.run(docs), text


def load_models():
    model = 'google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model)
    #model = AutoModelForSeq2SeqLM.from_pretrained(model, load_in_8bit=True)

    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=100
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


# Streamlit app
def main():
    llm = load_models()

    st.title("Webpage Text Summarizer and Question Answering")
    st.write("Enter a URL and get a summary of the webpage text. You can also ask questions about the content.")

    example_url = "https://www.abc.net.au/news/2023-07-11/crown-resorts-450-million-fine-money-laundering/102588950"
    st.markdown(
        f"For instance, you can copy the following example URL:<br>[`{example_url}`]({example_url})",
        unsafe_allow_html=True
    )

    # Input URL
    url = st.text_input("Enter URL:")
    if url:
        
        summary,text = summarize_text(url, llm)
        st.subheader("Summary:")
        st.write(summary)

        

if __name__ == "__main__":
    main()