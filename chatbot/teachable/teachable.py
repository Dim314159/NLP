
import pandas as pd
import nltk
import numpy as np
import re
import random
from nltk.stem import wordnet #lemmatization
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer #bow
from sklearn.feature_extraction.text import TfidfVectorizer #tfidf
from sklearn.metrics import pairwise_distances #cosine sim


lema = wordnet.WordNetLemmatizer()
def text_lemmatize(text):
    text_lower = str(text).lower() #to lower
    text_clean = re.sub(r'[^ a-z0-9]', '', text_lower) #cleaning
    replacement(text_clean, dict_replacement) #simplification
    tokens = nltk.wordpunct_tokenize(text_clean) #tokenizing
    tokens_and_tags = pos_tag(tokens, tagset = None) #pairs word-pos
    lemas_of_words = []
    
    for token, tag in tokens_and_tags:
        if tag.startswith('V'): #verb
            new_tag = 'v'
        elif tag.startswith('J'): #adjective
            new_tag = 'a'
        elif tag.startswith('R'): #adverb
            new_tag = 'r'
        else:
            new_tag = 'n' #noun
        token_lema = lema.lemmatize(token, new_tag) #lemmatization
        lemas_of_words.append(token_lema)
    separator = ' '
    return separator.join(lemas_of_words)


#replace particular words
dict_replacement = {'me':'i','myself':'i','my':'i','mine':'i',
           'us':'we','ourselves':'we','our':'we','ours':'we',
           'your':'you','yours':'you','yourself':'you','yourselves':'you',
           'her':'she','hers':'she','herself':'she',
           'him':'he','his':'he','himself':'he',
           'its':'it','itself':'it',
           'them':'they','their':'they','theirs':'they','themselves':'they'
}
def replacement(text, my_dict):
    for key, value in my_dict.items():
        text = text.replace(key, value)



def find_max_val_index(my_list, threshold):
    n = len(my_list)
    indices = []
    for i in range(n):
        if my_list[i] == threshold:
            indices.append(i)
        elif my_list[i] >= threshold:
            indices.clear()
            threshold = my_list[i]
            indices.append(i)
    n = len(indices)
    if n == 0:
        return 0, 0
    else:
        return threshold, random.choice(indices) 




def add_record(my_df, context_new, lemmitized_new, response_new):
    new_record = {'context':[context_new], 'lemmatized':[lemmitized_new], 'response':[response_new]}
    new_record_df = pd.DataFrame(new_record)
    return pd.concat([my_df, new_record_df], ignore_index=True)


# import data

df = pd.read_csv('data.csv')


# Initializing threshold from 0 to 1 for similarity,
# and method for processing: 'tfidf' for TF-IDF and 'cv' for Bag of Words

threshold = 0.7
method_flag = 'tfidf'


if method_flag == 'tfidf':
    method = TfidfVectorizer() # intializing tf-id
elif method_flag == 'cv':
    method = CountVectorizer() # intializing the count vectorizer

my_input = input('Talk(Enter to exit): ')
while True:
    if not my_input:
        break
    
    input_lemmatized = text_lemmatize(my_input)
    context_all_lemmatized = df['lemmatized'].tolist()
    
    response_indices = [i for i, x in enumerate(context_all_lemmatized) if x == input_lemmatized]
    if len(response_indices):
        response_index = random.choice(response_indices)
        response = df.at[response_index, 'response']
        print('Tama: ', response)
    else:
        context_all_transformed = method.fit_transform(context_all_lemmatized).toarray() # responses to tf-idf
        context_all_transformed_df = pd.DataFrame(context_all_transformed) 
        input_transformed = method.transform([input_lemmatized]).toarray() # applying tf-idf

        #features = method.get_feature_names()
        #df_similarity = pd.DataFrame(df, columns = ['response'])
        #df_similarity['similarity'] = cosine_value

        cosine_value = 1 - pairwise_distances(context_all_transformed_df, input_transformed, metric = 'cosine') #calculate similarity

        value_max, index_max = find_max_val_index(cosine_value, threshold)

        if value_max:
            response = df.at[index_max, 'response']
            print('Tama: ', response)
        else:
            print('Tama: I do not know what to say')
            new_response = input('Teach me(Enter to exit): ')
            if not new_response:
                break
            df = add_record(df, my_input, input_lemmatized, new_response)
            my_input = input('Talk(Enter to exit): ')
            continue
                
    current_input = my_input
    my_input = input('Talk(Enter to exit): ')
    
    if my_input == '!':
        print('Tama: Sorry, what is the correct response?')
        new_response = input('Teach me(Enter to exit): ')
        if not new_response:
            break
        df = add_record(df, current_input, input_lemmatized, new_response)
        my_input = input('Talk(Enter to exit): ')
        continue
        
    if input_lemmatized not in context_all_lemmatized:
        df = add_record(df, current_input, input_lemmatized, response)
    
    if my_input == '+':
        print('Tama: What else can you tell me about it?')
        new_response = input('Teach me(Enter to exit): ')
        if not new_response:
            break
        df = add_record(df, current_input, input_lemmatized, new_response)
        my_input = input('Talk(Enter to exit): ')
                
df.to_csv('data.csv', index = False)