# -*- coding: utf-8 -*-

"""
Created on Thursday, May 26, 2022 
Script para criação de aplicação de processamento de texto
@author: Prof. Leandro Romualdo da Silva
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import streamlit as st
import spacy 

nlp = spacy.load('en_core_web_sm')

try:
    from functions import *

    # app design
    app_meta('📊')
    set_bg_hack('image2.jpg')

    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.markdown("""---""")
    

    # Começamos o desenvolvimento do nosso app
    #st.title("Processamento de textos com Machine Learning.")
    new_title = '<p style="font-family:sans-serif; color:Black; font-size: 52px;">Processamento de linguagem natural com Machine Learning.</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    # aqui definimos alguns dos elementos do front end da página da web, como
    # a fonte e a cor de fundo, o preenchimento e o texto a ser exibido
    html_temp = """
    <div style ="background-color:blue;padding:13px">
    <h1 style ="color:white;text-align:center;">Aplicação de NLP - Turma 35 Inteligencia artificial Uninove </h1>
    </div>
    """

    # esta linha nos permite exibir os aspectos do front end que temos
    # definido no código acima
    st.markdown(html_temp, unsafe_allow_html = True)


    # esta é a função principal com a qual definimos nossa página da web
    def main():

        # Opções de predição
        subheader = '<p style="font-family:sans-serif; color:Black; font-size: 32px;">Selecione uma das opções abaixo: </p>'
        st.markdown(subheader, unsafe_allow_html=True)
        #st.subheader('** Selecione uma das opções abaixo:**')
        options = st.radio('O que deseja fazer?', ('','Classificação de textos', 'Geração de textos', 'Sumarização de textos', 
                                                   'Reconhecimento de entidades nomeadas', 'Análise de sentimento'))

        if options == 'Classificação de textos':
        # as linhas a seguir criam caixas de texto nas quais o usuário pode inserir
            # os dados necessários para fazer a previsão
            text=st.text_input("Digite aqui seu texto aqui:")
            result =" "
            # a linha abaixo garante que quando o botão chamado Predict for clicado,
            # a função de predict definida acima é chamada para fazer a previsão
            # e armazene-o no resultado variável
            if st.button("Predict"):
                result = prediction_classifier(text)
                st.text('Aguarde... Classificando texto...')
            st.success('O texto inserido foi classificado como {}:\n e com as seguintes entidades nomeadas : {}, {}'.format(result, nlp(text).ents[0].text, nlp(text).ents[0].label_,nlp(text).ents[1].text, nlp(text).ents[1].label_))

        # Chamada da função de geração de textos
        if options == 'Geração de textos':

            text, qtd = st.text_input("Digite um tema aqui:"), st.text_input("Quantidade de palavras a serem geradas:")
            result=""
            # Chamada da função para gerar textos
            if st.button("Gerar texto"):
                result=text_generated(text, qtd)
                st.text('Aguarde.... Gerando texto.')
            st.success(result)

        # Chamada da função de sumarização de textos
        if options == 'Sumarização de textos':
            result=""
            text, percent =  st.text_input('Digite aqui seu texto aqui:'), st.text_input('Digite aqui a porcentagem de textos a serem gerados:')
            if st.button("Sumarizar texto"):
                result = summarize(text, percent)
                st.text('Aguarde.... Sumarizando texto.')
            st.success(result)

        # Chamada da função de reconhecimento de entidades nomeadas
        if options == 'Reconhecimento de entidades nomeadas':
            result=""
            text = st.text_input('Digite aqui seu texto aqui:')
            if st.button("Reconhecer entidades nomeadas"):
                if text is not None:
                    result = get_ner(text)
                    st.text('Aguarde.... Analisando texto.')
            st.success(result)

        # Chamada da função de análise de sentimentos
        if options == 'Análise de sentimento':
            result=""
            text = st.text_input('Digite aqui seu texto aqui:')
            if st.button("Analisar sentimento"):
                if text is not None:
                    result = sentiment_analysis(text)
                    st.text('Aguarde.... Analisando texto.')
            st.success(result)

except ValueError:
    st.error("Oops, falha ao carregar o arquivo. Por favor, tente novamente.")

if __name__=='__main__':
    main()