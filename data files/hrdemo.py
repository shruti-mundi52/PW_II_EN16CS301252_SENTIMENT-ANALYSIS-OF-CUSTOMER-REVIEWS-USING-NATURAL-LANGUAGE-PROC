# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:49:22 2020

@author: Lenovo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data=pd.read_csv("hotel_review_final.csv")

data_new=data.iloc[:,[4,6,7,8]]

data_new['Negative_Review']=data_new["Negative_Review"].apply(lambda x:x.replace("No Negative"," "))

import nltk
from nltk.corpus import wordnet

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
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

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


data_new["Negative_Review"] = data_new["Negative_Review"].apply(lambda x: clean_text(x))


df=data_new.groupby("Hotel_Name")


hotel_no=data_new.Hotel_Name.nunique()
hotel_list=data_new['Hotel_Name'].unique()

import nltk
nltk.download('punkt')
from nltk import word_tokenize
def final_cat_func(df,hotel_list):
    lst=[]
    lst_name=[]
    for hotel in hotel_list:
        df_group=df.get_group(hotel)
        review=df_group['Negative_Review']
        str_rev=" "
        for i in review:
            str_rev=str_rev+i+" "
        lst.append(str_rev)
        lst_name.append(hotel)
    return lst,lst_name




lst,lst_name=final_cat_func(df,hotel_list)
lst=np.array(lst).reshape(-1,1)
lst_name=np.array(lst_name).reshape(-1,1)
df_lst=pd.DataFrame(lst,columns=['review'])
df_lstname=pd.DataFrame(lst_name,columns=['hotel_name'])
df_final=pd.concat([df_lstname,df_lst], axis=1)

from nltk.probability import FreqDist

fdist2=FreqDist()

def word_count(df_final):
    final_list=[]
    review=df_final['review']
    for i in review:
        str_demo=" "
        demo_list=[]
        str_demo=str_demo+i+" "
        demo_list.append(str_demo)
        for i in range(len(demo_list)):
            tok=word_tokenize(demo_list[i])
        final_list.append(tok)
    hotel_dict={}
    for i,n in zip(hotel_list,final_list):
        hotel_dict[i]=n
    return hotel_dict
    
hotel_dict=word_count(df_final)


def common_words_(hotel_dict,hotel_name):
    fdist=FreqDist()
    for k,v in hotel_dict.items():
        if k==hotel_name:
            val=hotel_dict.get(k)
            for words in val:
                fdist[words]+=1
            print(fdist.most_common(20))
common_words_(hotel_dict, "Park Plaza County Hall London" )
            

def common_words_list(hotel_dict,hotel_name):
    fdist=FreqDist()
    lst_freq=[]
    for k,v in hotel_dict.items():
        if k==hotel_name:
            val=hotel_dict.get(k)
            for words in val:
                fdist[words]+=1
            #print(fdist.most_common(20))
            lst_freq=fdist.most_common(20)
    return lst_freq
            
lst_freq=common_words_list(hotel_dict, "Park Plaza County Hall London" )


def create_record_freq(hotel_list,hotel_dict):
    lst_freq=[]
    lst_freq_final=[]
    for hotel in hotel_list:
        lst_freq=common_words_list(hotel_dict,hotel)
        lst_freq_final.append(lst_freq)
    return lst_freq_final
lst_freq_final=create_record_freq(hotel_list,hotel_dict)

lst_freq_final=np.array(lst_freq_final).reshape(-1,1)
hotel_list=np.array(hotel_list).reshape(-1,1)
lst_fd=pd.DataFrame(lst_freq_final,columns=['hotel_negative_review'])
lst_hotel=pd.DataFrame(hotel_list,columns=["hotel_name"])
data_record=pd.concat([lst_hotel,lst_fd],axis=1)

        





















