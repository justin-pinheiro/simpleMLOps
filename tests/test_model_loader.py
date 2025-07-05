import pytest
from unittest.mock import patch, MagicMock
from domain.iris_features import IrisFeatures
from app.model_loader import ModelLoader

def test_load_model_opens_file_and_loads_pickle():
    with patch("builtins.open"), patch("pickle.load") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        model = ModelLoader.load_model()
        
        mock_load.assert_called_once()
        assert model == mock_model

def test_predict_calls_model_and_gets_class_name():
    features = IrisFeatures(sepal_length=1.0, sepal_width=2.0, petal_length=3.0, petal_width=4.0)
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [42]
    
    with patch("app.model_loader.DataLoader.get_class_name") as mock_get_class_name:
        mock_get_class_name.return_value = "setosa"
        
        result = ModelLoader.predict(features, mock_model)
        
        mock_model.predict.assert_called_once_with([[1.0, 2.0, 3.0, 4.0]])
        
        mock_get_class_name.assert_called_once_with(42)
        
        assert result == "setosa"
