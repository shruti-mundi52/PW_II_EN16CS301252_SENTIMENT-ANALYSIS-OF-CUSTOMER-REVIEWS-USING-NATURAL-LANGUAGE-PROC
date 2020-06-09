# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:00:00 2020

@author: VAISHNAVI
"""

import pandas as pd
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
pos['review'] = data['Positive_Review']
pos['review']=pos["review"].apply(lambda x:x.replace("No Positive"," "))
pos['label'] = 0
neg['review']= data['Negative_Review']
neg['review']=neg["review"].apply(lambda x:x.replace("No Negative"," "))
neg['label'] = 1

#frames = [pos,neg]
#df = pd.concat(frames)

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
    
#df["review_clean"] = df["review"].apply(lambda x: clean_text(x))
pos["review_clean"] = pos["review"].apply(lambda x: clean_text(x))
neg["review_clean"] = neg["review"].apply(lambda x: clean_text(x))


#df['review_list'] = df['review_clean'].apply(lambda x : x.split())
pos['review_list'] = pos['review_clean'].apply(lambda x : x.split())
neg['review_list'] = neg['review_clean'].apply(lambda x : x.split())


#lst = []
#df['review_clean'].apply(lambda x: lst.append(x))
pos_lst = []
pos['review_clean'].apply(lambda x: pos_lst.append(x))
neg_lst = []
neg['review_clean'].apply(lambda x: neg_lst.append(x))


from gensim import corpora
#texts = [text.split() for text in lst]
pos_texts = [text.split() for text in pos_lst]
neg_texts = [text.split() for text in neg_lst]

#for text in texts:
#    for word in text:
#        word = word.encode('utf-8')
        

#dictionary = corpora.Dictionary(texts)
#corpus = [dictionary.doc2bow(text) for text in texts]

pos_dictionary = corpora.Dictionary(pos_texts)
pos_corpus = [pos_dictionary.doc2bow(text) for text in pos_texts]

neg_dictionary = corpora.Dictionary(neg_texts)
neg_corpus = [neg_dictionary.doc2bow(text) for text in neg_texts]

from gensim import models
#model = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
#
#topics = model.print_topics(num_words=3)
#for topic in topics:
#    print(topic)
    
pos_model = models.ldamodel.LdaModel(pos_corpus, num_topics=5, id2word=pos_dictionary, passes=5)

pos_topics = pos_model.print_topics(num_words=5)
print("Positive Topics are:")
for topic in pos_topics:
    print(topic)

    
neg_model = models.ldamodel.LdaModel(neg_corpus, num_topics=5, id2word=neg_dictionary, passes=5)

neg_topics = neg_model.print_topics(num_words=5)
print("Negative Topics are:")
for topic in neg_topics:    
    print(topic)

