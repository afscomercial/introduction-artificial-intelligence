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
        # Add a batch dimension to the input
        test_sample_expanded = np.expand_dims(data[0], axis=0)
        # Make prediction
        y_pred = (self.model.predict(test_sample_expanded) > 0.5).astype("int32")
        # Get the predicted class index
        predicted_index = y_pred[0][0]
        # Map the predicted index to the class label
        predicted_class = self.labels[predicted_index]

        return predicted_class
        
model_path = os.getenv('MODEL_PATH')

model_loader = ModelLoader(
        path=model_path,
        name='fatal_or_not_fatal', 
        version=1.0, 
        labels=['Non-Fatal Injury', 'Fatal']
)
