# %matplotlib inline
# from IPython import display

import os
import json
from flask import Flask, request, abort
from flask_cors import CORS

FILENAME = "/Users/ivan/Desktop/actions/arctic/project_sa/web_app.json" if "AMVERA" in os.environ else "web_app.json"

import io
import streamlit as st

import matplotlib
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import random

import pandas as pd
import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import os #new
from crypt import methods
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv, find_dotenv # new

load_dotenv(find_dotenv()) # new

# from tqdm.notebook import tqdm

from sklearn.impute import SimpleImputer

from flask import Flask, render_template # new

app = Flask(__name__)

@app.route('/')
# def index():
#     try:

#     return render_template('index.html') # new

# if __name__ == '__main__':
#     app.run()

@st.cache(allow_output_mutation=True)

# создаем модель машинного обучения 
def load_model():
    model = torch.load('mlp3.pth') # своя обученная глубокая модель
    return model


def load_dataset():
    """Создание формы для загрузки датасетов или предсказания"""
    # Форма для загрузки  средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите датасет для предсказания')
    if uploaded_file is not None:
        # Получение загруженного 
        download_data = uploaded_file.getvalue()
        # Показ загруженного  на Web-странице средствами Streamlit
        st.table(download_data)
        # Возврат  в формате PIL
        return uploaded_file.open(io.BytesIO(download_data))
    else:
        return render_template('error.html')


def print_predictions(preds):
    return st.write(print(model.predict(preds)))

# @app.route('/predict',methods=['POST'])
#def predict(preds):
#     #For rendering results on HTML GUI
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
    #print(model.predict())
#     output = round(prediction[0], 2) 
#     return render_template('index.html', prediction_text='CO2    Emission of the vehicle is :{}'.format(output))

# Загружаем предварительно обученную модель
model = load_model()

# Выводим заголовок страницы средствами Streamlit     
st.title('Прогноз ущерба, анализ риска')
# Вызываем функцию создания формы загрузки изображения
d_set = load_dataset()


# Показывам кнопку для запуска Создания прогноза
result = st.button('Создать прогноз')
# Если кнопка нажата, то запускаем 
if result:
    # # Предварительная обработка 
    x = load_dataset()
    # Распознавание изображения
    preds = model.predict(x)
    # Выводим заголовок результатов жирным шрифтом
    # используя форматирование Markdown
    #return render_template('index.html')
    st.write('**Результаты расчетов:**')
    # Выводим результаты 
    print_predictions(preds)