"""
Source: https://github.com/soft-nougat/dqw-ivves
@author: TNIKOLIC

"""
import json
import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import pycaret as pyc

import io
from PIL import Image
from pprint import pprint
from zipfile import ZipFile
import os
from os.path import basename

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Sumarização de textos
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Função de geração de novos textos
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
model = AutoModelWithLMHead.from_pretrained("pierreguillou/gpt2-small-portuguese")

# Análise de sentimento
from transformers import pipeline
classifier = pipeline("sentiment-analysis")


# Classificação de imagens
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from fastai.vision import *
    
model_class = VGG16(weights='imagenet')



def app_section_button(option1, option2, option3, option4):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # current page
        col1.markdown(option1) 
    with col2:
        st.markdown(option2, unsafe_allow_html=True) 
    with col3:
        st.markdown(option3, unsafe_allow_html=True) 
    with col4:
        st.markdown(option4, unsafe_allow_html=True) 
      

def app_meta(icon):

    # Set website details
    st.set_page_config(page_title ="Processamento de Linguagem natural", 
                       page_icon=icon, 
                       layout='centered')
    
    # set sidebar width
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# definindo a função que fará a previsão usando os dados que o usuário insere
def prediction_classifier(docs_new):

    docs_new = [docs_new]
    # carregar o modelo treinado
    model = joblib.load('mlp_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    docs = list(docs_new)
    #vectorizer = TfidfVectorizer()   
    X_new_tfidf_vectorize = vectorizer.transform(docs)

    sgd_predicted = model.predict(X_new_tfidf_vectorize)

    categories = ['rec.motorcycles', 'rec.autos']
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

    for doc, category in zip(docs_new, sgd_predicted):
        return twenty_train.target_names[category] 

# Função de sumarização de textos. Recebe um texto e um percentual que deve ser retornado
def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

# Função para geração de textos
def text_generated(input, qtd):

    tokenizer.model_max_length=1024 
    text = input
    inputs = tokenizer(text, return_tensors="pt")
    sample_outputs = model.generate(inputs.input_ids,
                    pad_token_id=50256,
                    do_sample=True, 
                    max_length=50, # put the token number you want
                    top_k=40,
                    num_return_sequences=1)
    # generated sequence
    for i, sample_output in enumerate(sample_outputs):
        return ">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist()))

# função de análise de sentimento
def sentiment_analysis(text):
    response = classifier(text)
    return response

# Função de classsificação de imagens usando VGG16
def image_classifier(file):
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    pred = model_class.predict(image)
    label = decode_predictions(pred)
    label = label[0][0]
    return '%s (%.2f%%)' % (label[1], label[2]*100)

# Função de extração de entidades nomeadas
def get_ner(text):
    ner = pipeline("ner", grouped_entities=True)
    result = ner(text)
    return result