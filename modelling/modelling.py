import pandas as pd
import numpy as np 
import pickle 
import yaml
from sklearn.ensemble import RandomForestClassifier

# Load parameters from params.yaml
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)
# Extract the number of estimators and max depth for RandomForest
n_estimators = params['modelling']['n_estimators']
max_depth = params['modelling']['max_depth']

train_data = pd.read_csv("data/interim/train_bow.csv")

x_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(x_train, y_train)

pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
