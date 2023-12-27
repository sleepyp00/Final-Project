import numpy as np 
import pandas as pd
import hopsworks
import torch
from torch.utils.data import TensorDataset, DataLoader
import json
import matplotlib.pyplot as plt
from utils import train_ensemble_standard, train_model, accuracy, get_dataloaders
import joblib

import pandas as pd
df = pd.read_json('Data/News_Category_Dataset_v3.json', lines = True)
df = df.drop_duplicates(subset='headline')
df = df.drop(columns=['link', 'headline', 'authors', 'date'])
unique_values = df['category'].unique()

# Convert the unique values to a list
unique_values_list = list(unique_values)

category_to_index = {
    "POLITICS":0,
    "TECH":1,
    "SCIENCE":1,
    "ENTERTAINMENT":2,
    "SPORTS":3,
    "BUSINESS":4,
    "MONEY":4,
}

index_to_category = {
    0:"politics",
    1:"science",
    2:"entertainment",
    3:"sports",
    4:"business"
}

value_counts = df['category'].value_counts()
summaries = df['short_description'].values
categories = df['category'].values
targets = np.zeros(len(categories), dtype=int)
for i, value in enumerate(categories):
    if value in category_to_index:
        targets[i] = category_to_index[value]
    else:
        targets[i] = -1

unique, counts = np.unique(targets, return_counts=True)

summaries = summaries[targets != -1]
categories = categories[targets != -1]
targets = targets[targets != -1]

#translate categories
categories = []
for target in targets:
    categories.append(index_to_category[target])

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()
embedding_model = mr.get_model("news_embedding", version = 1)
model_dir = embedding_model.download()
embedding_model = joblib.load(model_dir + "/news_embedding.pkl")

features = embedding_model.encode(summaries, show_progress_bar=True)

data = {"embedding":features.tolist(),
        "target":targets.tolist(), 
        "category":categories}

df = pd.DataFrame(data)
df.to_csv('output.csv', index=False)

df = pd.read_csv('output.csv')

columns = df.columns.tolist()

dataset_api = project.get_dataset_api()
dataset_api.upload("output.csv", "Resources/FinalProject", overwrite=True)



""" df_1 = df.iloc[:10000,:]
df_2 = df.iloc[10000:20000,:]
df_3 = df.iloc[20000:30000,:]
df_4 = df.iloc[30000:40000,:]
df_5 = df.iloc[40000:50000,:]
df_6 = df.iloc[50000:60000,:]
df_7 = df.iloc[60000,:]

def insert_df(df):
    news_fg = fs.get_or_create_feature_group(
        name="base_category_dataset",
        version=1,
        primary_key=columns,
        description="Base news category dataset")
    news_fg.insert(df)

insert_df(df_1)
insert_df(df_2)
insert_df(df_3)
insert_df(df_4)
insert_df(df_5)
insert_df(df_6)
insert_df(df_7) """




pass

""" N = len(summaries)
split_ind = round(N*0.9)
summaries_train, summaries_val = summaries[:split_ind], summaries[split_ind:] 
targets_train, targets_val = targets[:split_ind], targets[split_ind:] 




datasets = {
    "train":TensorDataset(summaries_train, targets_train),
    "validation":TensorDataset(summaries_val, targets_val)
}
pass """