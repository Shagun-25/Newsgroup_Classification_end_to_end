import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
import os
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree

nltk.download("popular")
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class ColumnsPreprocessor(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self):
        pass

    def decontracted(self, phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    
    #https://stackoverflow.com/questions/48660547/how-can-i-extract-gpelocation-using-nltk-ne-chunk
    def get_continuous_chunks(self, text, label):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        prev = None
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if type(subtree) == Tree and subtree.label() == label:
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk
        
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        X = X.reset_index(drop=True)

        X['text']=X['text'].apply(str)
        
        try:
            #Extracting domain names from emails
            for row in tqdm(X.itertuples()):
                match = re.findall(r'[\w\.-]+@[\w\.-]+', row.text)
                emails = []
                for email in match:
                    lst = email.split('@')[1].split(".")
                    lst1 = lst.copy()
                    for word in lst1:
                        if len(word) <= 2 or word == 'com':
                            lst.remove(word)
                    emails.extend(lst)
                emails = " ".join(list(set(emails)))
                X.at[row[0], 'preprocessed_email'] = emails

            #Removing emails
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'[\w\.-]+@[\w\.-]+', " ", row.text)

            #Extracting Subjets
            for row in tqdm(X.itertuples()):
                match = re.findall(r'(Subject:+(.*?)+\n)', row.preprocessed_text)
                match1 = match[0][0]
                subject = match1.split(':')[-1].split("\n")[0]
                subject1 = re.sub('[^A-Za-z0-9 ]+', ' ', subject)
                subject2 = re.sub(r'(?i)\b[a-z]\b', "", subject1)
                X.at[row[0], "preprocessed_subject"] = subject2
                X.at[row[0], 'preprocessed_text'] = re.sub(r'(Subject:+(.*?)+\n)', " ", row.preprocessed_text)

            #Removing sentences starting with "Write to:" or "From:" or "Newsgroups:" or "E-Mail :" or "Reply-To:" or "Sender:" or "Xref:" or "Path:" or "Message-ID:" or "References:"
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'((From:|Write to:|Newsgroups:|E-Mail :|Reply-To:|Sender:|Xref:|Path:|Message-ID:|References:)+(.*?)+\n)', " ", row.preprocessed_text)

            #Removing all the words between < >
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'<([^>]+)>', "", row.preprocessed_text)

            #Removing all the words between ( )
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'\(([^)]+)\)', "", row.preprocessed_text)

            #Removing all the newlines ('\n'), tabs('\t'), "-", "\".
            for row in tqdm(X.itertuples()):
                string = X.at[row[0], 'preprocessed_text'].replace('\n', ' ').replace('\t', ' ').replace('-', ' ').replace('\\', ' ')
                X.at[row[0], 'preprocessed_text'] = string

            #Removing all the words which ends with ":".
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'\w*:', " ", row.preprocessed_text)

            #Decontracting the text
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = self.decontracted(X.at[row[0], 'preprocessed_text'])

            #Replacing all the digits with space
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'[\d]', " ", row.preprocessed_text)
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'[\'"]', " ", row.preprocessed_text)

            #Chunking the data
            #Removing Locations and Person Names from the text

            for row in tqdm(X.itertuples()):
                gpe = self.get_continuous_chunks(row.preprocessed_text, 'GPE')
                person = self.get_continuous_chunks(row.preprocessed_text, 'PERSON')

                my_sent = row.preprocessed_text

                for word in person:
                    if len(word.split()) == 1:
                        my_sent= my_sent.replace(word, " ")
                    else:
                        for x in word.split():
                            my_sent= my_sent.replace(word, " ")

                for place in gpe:
                    if len(place.split()) != 1:
                        my_sent= my_sent.replace(" ".join(place.split()), "_".join(place.split()))
                    else:
                        pass
                
                X.at[row[0], 'preprocessed_text'] = my_sent

            #Removing '_' from '_word' or 'word_'
            for row in tqdm(X.itertuples()):
                mystr = row.preprocessed_text
                mystr1 = re.sub(r' _', " ", mystr)
                mystr2 = re.sub(r'_ ', " ", mystr1)
                mystr3 = re.sub(r'_$', " ", mystr2)
                mystr4 = re.sub(r'^_', " ", mystr3)
                X.at[row[0], 'preprocessed_text'] = mystr4

            #Removing 'Oneletter_' and 'Twoletter_' words
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r' \w{1,2}_', " ", row.preprocessed_text)

            #Replacing all the words except "A-Za-z_" with space.
            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_text'] = re.sub(r'[^A-Za-z_]', " ", row.preprocessed_text)

            #Converting in lower case and removing words with length >= 15 and <= 2
            for row in tqdm(X.itertuples()):
                mystr = row.preprocessed_text
                mystr = mystr.lower()
                mystr_list = mystr.split()
                x = mystr_list.copy()
                for word in mystr_list:
                    if len(word) >= 15 or len(word) <= 2:
                        x.remove(word)
                    else:
                        pass
                joined_mystr = " ".join(x)
                X.at[row[0], 'preprocessed_text'] = joined_mystr
                X.at[row[0], 'preprocessed_subject'] = row.preprocessed_subject
                X.at[row[0], 'preprocessed_email'] = row.preprocessed_email

            #Removing trailing spaces and multiple spaces
            for row in tqdm(X.itertuples()):
                mystr = row.preprocessed_text
                mystr = mystr.strip()
                X.at[row[0], 'preprocessed_text'] = re.sub(' +', ' ', mystr)

            for row in tqdm(X.itertuples()):
                X.at[row[0], 'preprocessed_subject'] = str(row.preprocessed_subject).strip()
                X.at[row[0], 'preprocessed_email'] = str(row.preprocessed_email).strip()

            #Combining preprocessed_text ,preprocessed_subject, preprocessed_email
            for i in range(len(X)):
                X.at[i, 'total_preprocessed_text'] = X.at[i, 'preprocessed_email'] + " " + X.at[i, 'preprocessed_subject'] + " " + X.at[i, 'preprocessed_text']
            
            X['count'] = X['total_preprocessed_text'].apply(lambda a: len(a.split()))
            X = X[['total_preprocessed_text','count']]

            return X

        except Exception as e:
            raise CustomException(e,sys)
        
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            text_columns = ["text"]

            text_pipeline=Pipeline(

                steps=[
                ("columnpreprocessor",ColumnsPreprocessor())
                ]
            )

            logging.info(f"Text columns: {text_columns}")

            preprocessor=ColumnTransformer(
                [
                ("text_pipeline",text_pipeline, ['filename','text','preprocessed_text','preprocessed_email','preprocessed_subject'])
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="class"
            text_columns = ["text"]

            d = {0: 'alt.atheism', 1: 'comp.graphics', 2: 'comp.os.ms-windows.misc', 3: 'comp.sys.ibm.pc.hardware', 4: 'comp.sys.mac.hardware', 5: 'comp.windows.x', 6: 'misc.forsale', 7: 'rec.autos', 8: 'rec.motorcycles', 9: 'rec.sport.baseball', 10: 'rec.sport.hockey', 11: 'sci.crypt', 12: 'sci.electronics', 13: 'sci.med', 14: 'sci.space', 15: 'soc.religion.christian', 16: 'talk.politics.guns', 17: 'talk.politics.mideast', 18: 'talk.politics.misc', 19: 'talk.religion.misc'}
            d = {v: k for k, v in d.items()}

            train_df = train_df.replace({"class": d})
            test_df = test_df.replace({"class": d})

            # train_df['class'] = train_df['class'].astype('category').cat.codes
            # test_df['class'] = test_df['class'].astype('category').cat.codes

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                d
            )
        except Exception as e:
            raise CustomException(e,sys)