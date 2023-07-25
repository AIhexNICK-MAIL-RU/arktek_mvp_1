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

# from tqdm.notebook import tqdm

from sklearn.impute import SimpleImputer


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
        st.image(download_data)
        # Возврат  в формате PIL
        return Image.open(io.BytesIO(download_data))
    else:
        return None


def print_predictions(preds):
    st.write(y_val)

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
    # x = preprocess_image(img)
    # Распознавание изображения
    preds = model.predict(x)
    # Выводим заголовок результатов жирным шрифтом
    # используя форматирование Markdown
    st.write('**Результаты расчетов:**')
    # Выводим результаты 
    print_predictions(preds)
