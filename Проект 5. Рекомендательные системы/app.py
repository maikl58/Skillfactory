import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse


def nearest_product_nms(itemid, index, n=10):
    nn = index.knnQuery(item_embeddings[itemid], k=n)
    return nn

# Чтение файла с названием товара

def read_files(folder_name='data'):
    """
    Функция для чтения файлов 
    """
    df = pd.read_csv(folder_name+'/meta_Grocery.csv')
    train = pd.read_csv(folder_name+'/train.csv')
    test = pd.read_csv(folder_name+'/test.csv')
    # Связываем asin и itemid
    df_asin = pd.concat([train[['asin', 'itemid']], test[['asin', 'itemid']]])
    mapper = dict(zip(df_asin.asin,df_asin.itemid))
    df['itemid'] = df.asin.apply(lambda x: mapper[x])
    return df
   
    return df
        
##

def load_embeddings():
    """
    Функция для загрузки векторных представлений
    """
    with open('./data/item_embeddings.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать наш быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx

 ##
def get_names(index):
    """
    
    Функция для возвращения названия товара и цены
    
    """
    names = []
    for idx in index:
        names.append('Product name:  {} '.format(
            name_mapper[idx]) + '  Product Price: {}'.format(author_mapper[idx]))
    return names

## 
def make_mappers():
    """
    Функция для создания отображения id в title
    """
    name_mapper = dict(zip(df.itemid, df.title))
    author_mapper = dict(zip(df.itemid, df.price))

    return name_mapper, author_mapper

  #Загружаем данные
df = read_files(folder_name='data') 
name_mapper, author_mapper = make_mappers()
item_embeddings,nms_idx = load_embeddings()

#Форма для ввода текста
title = st.text_input('Product Name', '')
#title = title.lower()


#Наш поиск по товарам
output = df[df.title.str.contains(title) > 0].head(10)

#Выбор товара из списка
option = st.selectbox('Which product?', output['title'].values)

#Выводим товар
'You selected: ', option

#Ищем рекомендации
val_index = output[output['title'].values == option].itemid
index = nearest_product_nms(val_index, nms_idx, 5)


#Выводим рекомендации к ней
'Most simmilar product are:'

st.write('', get_names(index[0])[1:])