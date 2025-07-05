from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from model_trainer import ModelTrainer
from model_saver import ModelSaver
from data_loader import DataLoader

from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    X, y = DataLoader.load_iris_data()
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    
    X_train, X_test, y_train, y_test = ModelTrainer.split_data(X, y)

    model = KNeighborsClassifier()

    trained_model = ModelTrainer.train_model(model, X_train, y_train)

    score = ModelTrainer.evaluate_model(trained_model, X_test, y_test)
    print(f"Model accuracy: {score:.3f}")

    ModelSaver.save(trained_model, "models/model.pkl")

    saved_model = ModelSaver.load("models/model.pkl")
    
    score = ModelTrainer.evaluate_model(saved_model, X_test, y_test)
    print(f"Model accuracy: {score:.3f}")