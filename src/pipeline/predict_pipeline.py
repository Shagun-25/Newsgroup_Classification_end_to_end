import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)

            save_directory = "E:/Data_Science/Newsgroup_Classification_end_to_end/artifacts/"
            tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)
            model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

            #predicting from the model

            predict_input = tokenizer_fine_tuned.encode(
                data_scaled[0].tolist()[0],
                truncation = True,
                padding = True,
                return_tensors = 'tf'    
            )

            output = model_fine_tuned(predict_input)[0]
            prediction_value = tf.argmax(output, axis = 1).numpy()[0]

            return prediction_value
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        text: str):

        self.text = text

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "text": [self.text]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)