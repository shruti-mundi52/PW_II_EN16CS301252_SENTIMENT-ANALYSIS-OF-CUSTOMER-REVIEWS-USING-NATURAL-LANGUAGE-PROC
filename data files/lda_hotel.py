# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:55:24 2020

@author: VAISHNAVI
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:00:00 2020

@author: VAISHNAVI
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

data = pd.read_csv('hotel_review_final.csv')
pos = pd.DataFrame()
neg = pd.DataFrame()

pos = data.iloc[:,[4,9,10,11]]
neg = data.iloc[:,[4,6,7,8]]

pos['Positive_Review']=pos["Positive_Review"].apply(lambda x:x.replace("No Positive"," "))
neg['Negative_Review']=neg["Negative_Review"].apply(lambda x:x.replace("No Negative"," "))


def get_wordnet_pos(pos_tag):
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
    
def clean_text(text):
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
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
    

pos["Positive_Review"] = pos["Positive_Review"].apply(lambda x: clean_text(x))
neg["Negative_Review"] = neg["Negative_Review"].apply(lambda x: clean_text(x))

pos_df = pos.groupby("Hotel_Name")
neg_df = neg.groupby("Hotel_Name")

hotel_no = data.Hotel_Name.nunique()
hotel_list = data['Hotel_Name'].unique()

import nltk
nltk.download('punkt')
from nltk import word_tokenize

def final_cat_func(df, hotel_list,label):
    lst=[]
    lst_name=[]
    for hotel in hotel_list:
        df_group = df.get_group(hotel)
        review = df_group[label]
        lst_rev=[]
        for i in review:
            lst_rev.append(i)
        lst.append(lst_rev)
        lst_name.append(hotel)
    return lst,lst_name


pos_lst, pos_lst_name = final_cat_func(pos_df,hotel_list, 'Positive_Review')
neg_lst, neg_lst_name = final_cat_func(neg_df,hotel_list, 'Negative_Review')

pos_texts = []
for lst in pos_lst:
    for text in lst:
        pos_texts.append(text.split())

neg_texts = []
for lst in neg_lst:
    for text in lst:
        neg_texts.append(text.split())


from gensim import corpora


pos_dictionary = corpora.Dictionary(pos_texts)
pos_corpus = [pos_dictionary.doc2bow(text) for text in pos_texts]

neg_dictionary = corpora.Dictionary(neg_texts)
neg_corpus = [neg_dictionary.doc2bow(text) for text in neg_texts]

from gensim import models
    
pos_model = models.ldamodel.LdaModel(pos_corpus, num_topics=5, id2word=pos_dictionary, passes=15)
pos_topics = pos_model.print_topics(num_words=15)
print("Positive Topics are:")
for topic in pos_topics:
    print(topic)

    
neg_model = models.ldamodel.LdaModel(neg_corpus, num_topics=5, id2word=neg_dictionary, passes=15)
neg_topics = neg_model.print_topics(num_words=15)
print("Negative Topics are:")
for topic in neg_topics:    
    print(topic)

