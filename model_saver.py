import pickle

class ModelSaver:
    
    @staticmethod
    def save(model, filename):
        """
        Save the model to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            
    @staticmethod
    def load(filename):
        """
        Load the model from a file.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
