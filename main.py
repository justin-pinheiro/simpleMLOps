from app.model_loader import ModelLoader
from domain.iris_features import IrisFeatures

from fastapi import FastAPI
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(features: IrisFeatures):
    model = ModelLoader.load_model()
    ModelLoader.predict(features, model)
