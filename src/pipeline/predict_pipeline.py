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
                data_scaled,
                truncation = True,
                padding = True,
                return_tensors = 'tf'    
            )

            output = model_fine_tuned(predict_input)[0]
            prediction_value = tf.argmax(output, axis = 1).numpy()[0]

            return prediction_value

            # model_path=os.path.join("artifacts","model.pkl")
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            # print("Before Loading")
            # model=load_object(file_path=model_path)
            # preprocessor=load_object(file_path=preprocessor_path)
            # print("After Loading")
            # data_scaled=preprocessor.transform(features)
            # preds=model.predict(data_scaled)
            # return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)