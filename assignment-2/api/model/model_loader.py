import typing
from pathlib import Path
import numpy as np
import tensorflow as tf

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
        


model_loader = ModelLoader(
        path='./my_model.keras', 
        name='fatal_or_not_fatal', 
        version=1.0, 
        labels=['Non-Fatal Injury', 'Fatal']
)


    
    #['INJURY', 'INVAGE', 'PASSENGER', 'SPEEDING', 'TRUCK', 'TRAFFCTL_No Control', 'LIGHT_Natural Light', 
    # 'LIGHT_Dark', 'ALCOHOL', 'TRAFFCTL_Automated Control', 'DISTRICT_Scarborough', 'DISTRICT_Toronto and East York', 
    # 'DISTRICT_Etobicoke York', 'TRSN_CITY_VEH', 'REDLIGHT', 'LIGHT_Artificial Light','DISTRICT_North York']
    
    # features = np.array([
        # 0 = Non-Fatal Injury
    #     [3., 2., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]
    # ])