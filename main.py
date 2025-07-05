import pickle
from typing import Union
from domain.iris_features import IrisFeatures
from app.data_loader import DataLoader

from fastapi import FastAPI
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(features: IrisFeatures):
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
        prediction = model.predict(input_data)
        return {"prediction": DataLoader.get_class_name(prediction[0])}
