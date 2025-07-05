from sklearn import datasets

class DataLoader:
    """
    A class to load datasets.
    """

    @staticmethod
    def load_iris_data():
        """
        Load the Iris dataset from sklearn and return the features and target.
        """
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        return X, y
    
    @staticmethod
    def get_class_name(class_index: int) -> str:
        """
        Get the class name for a given class index.
        """
        iris = datasets.load_iris()
        return iris.target_names[class_index]