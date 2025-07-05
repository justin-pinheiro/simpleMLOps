

import pickle

from app.data_loader import DataLoader
from domain.iris_features import IrisFeatures


class ModelLoader():
    
    @staticmethod
    def load_model():
        with open("models/model.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def predict(features: IrisFeatures, model):
        input_data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]
        
        prediction = model.predict(input_data)
        return DataLoader.get_class_name(prediction[0])
