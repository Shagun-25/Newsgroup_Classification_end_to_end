import os
import sys
from dataclasses import dataclass
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification

import tensorflow as tf

@dataclass
class ModelTrainer:
    def __init__(self):
        pass

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average = 'micro')
        precision = precision_score(y_true=labels, y_pred=pred, average = 'micro')
        f1 = f1_score(y_true=labels, y_pred=pred, average = 'micro')

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            X_train = [str(i) for i in X_train.tolist()]
            X_test = [str(i) for i in X_test.tolist()]
            y_train = np.asarray(y_train).astype('float32')
            y_test = np.asarray(y_test).astype('float32')

            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

            train_encodings = tokenizer(X_train, truncation = True, padding = True)
            test_encodings = tokenizer(X_test, truncation = True, padding = True)

            train_dataset = tf.data.Dataset.from_tensor_slices((
                dict(train_encodings),
                y_train
            ))

            test_dataset = tf.data.Dataset.from_tensor_slices((
                dict(test_encodings),
                y_test
            ))

            training_args = TFTrainingArguments(
                output_dir='./results',          
                num_train_epochs=3,              
                per_device_train_batch_size=16,  
                per_device_eval_batch_size=64,   
                warmup_steps=500,                
                weight_decay=1e-5,               
                logging_dir='./logs',            
                eval_steps=100                   
            )

            with training_args.strategy.scope():
                trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 20)

            trainer = TFTrainer(
                model=trainer_model,                 
                args=training_args,                  
                train_dataset=train_dataset,         
                eval_dataset=test_dataset,  
                compute_metrics=self.compute_metrics          
            )

            trainer.train()
            f1 = trainer.evaluate()

            #Saving the model
            save_directory = "E:/Data_Science/Newsgroup_Classification_end_to_end/artifacts/" 
            trainer_model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)

            return f1
            
        except Exception as e:
            raise CustomException(e,sys)