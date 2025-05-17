#!/usr/bin/env python
# coding: utf-8






    
# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import transformers
import torch
import io
import re
import requests
import pickle
import json
import spacy


from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric

import matplotlib.pyplot as plt
import matplotlib as mpl

#global constants
MODEL_DIRECTORY = "/srv/app/model/data/"

#model = SentenceTransformer('all-MiniLM-L6-V2')
#model.save(MODEL_DIRECTORY)
model = SentenceTransformer(MODEL_DIRECTORY)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

import os
for dirname, _, filenames in os.walk('/srv/notebooks/data/ESCU'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print(model)











    
# In[7]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df_query = pd.read_csv(f)
        #full_text = f.read()
        #for l in re.split(r"(\.)", full_text):
         #   if l != ".":
          #      name.append(l + "\n")
    #pd.DataFrame(string)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    nlp = spacy.load('en_core_web_sm')
    #dfs = df.applymap(str)
    sbd = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sbd)
    text = str(df_query.content.values)
    doc = nlp(text)
    sentence_list =[]
    for sentence in doc.sents:
        sentence_list.append(sentence.text)
    dfnew = pd.DataFrame({'content': sentence_list})
    return df_query, param, dfnew







    
# In[9]:


# initialize the model
# params: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = SentenceTransformer(MODEL_DIRECTORY)
    #return model
    #model = {}
    #model['hyperparameter'] = 42.0
    return model









    
# In[12]:


# returns a fit info json object
def fit(model,df,param):
    info = {"message": "model trained"}
    return info







    
# In[14]:


def apply(model,dfnew,param):
    model = SentenceTransformer(MODEL_DIRECTORY)
    escu_list_0 =[]
    escu_list_1 = []
    escu_list_2 = []
    escu_list_3 = []
    escu_list_4 = []
    df = pd.read_csv("/srv/notebooks/data/ESCU/ESCU_DataSource.csv", encoding='latin-1', usecols=['id_desc', 'datasource'])
    embedding_arr = model.encode(df['id_desc'])
    embedding_arr.shape
    for index, row in dfnew.iterrows():
        query_embedding = model.encode(str(row['content']))
        dist = DistanceMetric.get_metric('euclidean') # other distances: manhattan, chebyshev
        # compute pair wise distances between query embedding and all resume embeddings
        dist_arr = dist.pairwise(embedding_arr, query_embedding.reshape(1, -1)).flatten()
        # sort results
        idist_arr_sorted = np.argsort(dist_arr)
        escu_list_0.append(df['id_desc'].iloc[idist_arr_sorted[0]])
        escu_list_1.append(df['id_desc'].iloc[idist_arr_sorted[1]])
        escu_list_2.append(df['id_desc'].iloc[idist_arr_sorted[2]])
        escu_list_3.append(df['id_desc'].iloc[idist_arr_sorted[3]])
        escu_list_4.append(df['id_desc'].iloc[idist_arr_sorted[4]])
        #suggested_escu=df['id_desc'].iloc[idist_arr_sorted[0]]
    #print(spl_list)
    dfnew['suggested_escu_1']=escu_list_0
    dfnew['suggested_escu_2']=escu_list_1
    dfnew['suggested_escu_3']=escu_list_2
    dfnew['suggested_escu_4']=escu_list_3
    dfnew['suggested_escu_5']=escu_list_4

    return dfnew







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    # model will not be saved or reloaded as it is pre-built
    with open(MODEL_DIRECTORY + name + ".json", 'wb') as file:
        #model_dict=model.state_dict()
        #model_list=list(model)
         pickle.dump(model, file)
     #   torch.save(model.state_dict(), file)
    return model







    
# In[18]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    # model will not be saved or reloaded as it is pre-built
    #model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'rb') as file:
        model=pickle.load(file)
    #model = {}
     #   model = json.load(file)
    #model=model.load_model("t5","/srv/outputs/simplet5-epoch-4-train-loss-0.9567-val-loss-0.8704", use_gpu=False)
    return model





    
# In[19]:


# return model summary
def summary(model=None):
    returns = {"version": {"spacy": spacy.__version__} }
    if model is not None:
        s = []
        returns["summary"] = ''.join(s)
    return returns













