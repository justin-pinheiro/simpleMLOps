from sklearn.model_selection import train_test_split

class ModelTrainer:
    """A class to handle model training and evaluation."""

    @staticmethod
    def split_data(X, y, test_size=0.2):
        """
        Split the dataset into training and testing sets.
        """
        return train_test_split(X, y, test_size=test_size)

    @staticmethod
    def train_model(model, X_train, y_train):
        """
        Train the model on the training data.
        """
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate the model on the test data.
        """
        return model.score(X_test, y_test)
