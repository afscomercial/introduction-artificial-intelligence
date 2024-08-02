import typing
from pathlib import Path
import numpy as np
#from enum import Enum


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
        import tensorflow as tf
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
        



if __name__ == "__main__":
    
    #['INJURY', 'INVAGE', 'PASSENGER', 'SPEEDING', 'TRUCK', 'TRAFFCTL_No Control', 'LIGHT_Natural Light', 
    # 'LIGHT_Dark', 'ALCOHOL', 'TRAFFCTL_Automated Control', 'DISTRICT_Scarborough', 'DISTRICT_Toronto and East York', 
    # 'DISTRICT_Etobicoke York', 'TRSN_CITY_VEH', 'REDLIGHT', 'LIGHT_Artificial Light','DISTRICT_North York']
    
    features = np.array([
        # 0 = Non-Fatal Injury
        [3., 2., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]
    ])

    models_path = 'assignment-2\model\models\\tf\my_model.keras'

    model = ModelLoader(
        path=models_path, 
        name='fatal_or_not_fatal', 
        version=1.0, 
        labels=['Non-Fatal Injury', 'Fatal']
    )
    
    
    prediction = model.predict(features)
    print(prediction)
