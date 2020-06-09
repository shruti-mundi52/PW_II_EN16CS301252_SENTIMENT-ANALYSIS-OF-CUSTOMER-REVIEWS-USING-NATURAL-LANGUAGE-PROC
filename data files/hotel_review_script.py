# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:19:05 2020

@author: varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class hotel_review:
    
    def get_wordnet_pos(self,pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def clean_text(self,text):
    # lower text
        text = text.lower()
    # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
    # remove empty tokens
        text = [t for t in text if len(t) > 0]
    # pos tag text
        pos_tags = pos_tag(text)
    # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], self.get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
        text = [t for t in text if len(t) > 1]
    # join all
        text = " ".join(text)
        return(text)
    
    def merge_review(self,data):
        data["review"] = data["Negative_Review"] + ["Positive_Review"]
        return data

    def add_label(self,data):
        data["is_bad_review"] = data["Reviewer_Score"].apply(lambda x: 1 if x < 5 else 0)
        return data

    def select_columns(self,data):
        columns=["review","is_bad_review"]
        data_selected = data[columns]
        data_selected["review"] = data_selected["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
        return data_selected
    
    def sentiment_analysis(self,data):
        self.sid = SentimentIntensityAnalyzer()
        data["sentiments"] = data["review"].apply(lambda x: self.sid.polarity_scores(x))
        data = pd.concat([data.drop(['sentiments'], axis=1), data['sentiments'].apply(pd.Series)], axis=1)
        data=data.drop(['neu'],axis=1)
        return data
    
    def add_word_char(self,data):
        data["nb_chars"] = data["review"].apply(lambda x: len(x))
        data["nb_words"] = data["review"].apply(lambda x: len(x.split(" ")))
        return data
    
    def tf_idf(self,data):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(min_df = 10)
        self.t=self.tfidf.fit(data["review_clean"])
        tfidf_result = self.tfidf.transform(data["review_clean"]).toarray()
        tfidf_df = pd.DataFrame(tfidf_result, columns = self.tfidf.get_feature_names())
        tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
        tfidf_df.index = data.index
        data_final = pd.concat([data, tfidf_df], axis=1)
        return data_final
    
    def splitting(self,data):
        data=self.hotel_pipeline(data)
        X=self.X(data)
        Y=self.Y(data)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
        return X_train,X_test,y_train,y_test
        
    
    def X(self,data):
        self.label = "is_bad_review"
        ignore_cols = [self.label, "review", "review_clean"]
        features = [c for c in data.columns if c not in ignore_cols]
        X=data[features]
        return X
        
    def Y(self,data):
        Y=data[self.label]
        return Y

     
    def hotel_pipeline(self,data):
        data=self.merge_review(data)
        data=self.add_label(data)
        data=self.select_columns(data)
        data["review_clean"] = data["review"].apply(lambda x: self.clean_text(x))
        data=self.sentiment_analysis(data)
        data=self.add_word_char(data)
        data_final=self.tf_idf(data)
        return data_final

    def tfidf_input(self,data):
        tfidf_result =self.tfidf.transform(data["review_clean"]).toarray()
        tfidf_df = pd.DataFrame(tfidf_result, columns = self.tfidf.get_feature_names())
        tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
        tfidf_df.index = data.index
        data_final = pd.concat([data, tfidf_df], axis=1)
        return data_final
    
    def input_pipeline(self,li):
        data_=pd.DataFrame(li,columns=['review'])
        data_["review_clean"] = data_["review"].apply(lambda x: self.clean_text(x))
        data_=self.sentiment_analysis(data_)
        data_=self.add_word_char(data_)
        data_final=self.tfidf_input(data_)
        data_final=self.X(data_final)
        return data_final
    
        
    