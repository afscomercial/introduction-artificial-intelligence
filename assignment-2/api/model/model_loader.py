import os
import typing
from pathlib import Path
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

class ModelLoader(object):
    def __init__(self, 
                  path: Path, 
                  name: str, 
                  version: float = 1.0,
                  labels: typing.List[str] = None):
        self.path = path
        self.name = name
        self.version = version
        self.labels = labels
        self.model = self.__load_tensorflow_model()



    def __load_tensorflow_model(self):
        """"
        Load tensorflow model from path
        """
        model = tf.keras.models.load_model(self.path)
        return model
    
    

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict data using model
        """
        predictions =  self.model.predict(data)
        predictions = predictions.tolist()
        if self.labels:
            predictions = [self.labels[np.argmax(prediction)] for prediction in predictions]
        return predictions
        
model_path = os.getenv('MODEL_PATH')

model_loader = ModelLoader(
        path=model_path,
        name='fatal_or_not_fatal', 
        version=1.0, 
        labels=['Non-Fatal Injury', 'Fatal']
)
