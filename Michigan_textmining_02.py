
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[ ]:

import nltk
# nltk.download()
# from nltk.book import *
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# nltk.download('punkt')

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

print('wtf')

# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:



def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[ ]:

def answer_one():

    diversity = example_two() / example_one()
    assert type(diversity) == type(0.12134), 'is not float'

    return diversity

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[ ]:

def answer_two():
    from nltk.probability import FreqDist

    dist = FreqDist(text1)#.most_common() in ['whale', 'Whale']
    perc = (dist['whale'] + dist['Whale']) * 100 / len(nltk.word_tokenize(moby_raw))

    return perc

answer_two()


def answer_two():
    from nltk.probability import FreqDist

    words_lower = [w.lower() for w in text1]
    dist = FreqDist(words_lower)

    return dist['whale']/example_one() * 100


answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`.
# The list should be sorted in descending order of frequency.*

# In[ ]:


def answer_three():
    from nltk.probability import FreqDist
    # https: // docs.python.org / 2 / library / collections.html?highlight = most_common
    # collections.Counter.most_common
        # words = re.findall(r'\w+', open('hamlet.txt').read().lower())
        # Counter(words).most_common(10)
    return FreqDist(text1).most_common(20)  # sorted list of tuples

answer_three()


def answer_three():

    from nltk.probability import FreqDist
    # checkthis = [(w, f) for w, f in FreqDist(text1).items()]
    # checkthis.sort(key=lambda tup: tup[1], reverse=True)

    unique_tokens = set(nltk.word_tokenize(moby_raw))

    result = [(k, v) for k, v in FreqDist(text1).items()]
    result.sort(key=lambda tup: tup[1], reverse=True)
    # print(result[:20])

    dist = pd.DataFrame([FreqDist(text1)]).T
    dist.sort_values(by=0, ascending=False, inplace=True)
    tokens = list(zip(dist.head(20).index, dist.head(20)[0]))

    assert tokens == result[:20], 'unique test'
    assert len(dist2) == len(unique_tokens), str(len(dist2))+'is not'+str(len(unique_tokens))
    assert type(tokens) == type([]), 'type check'
    assert len(tokens) == 20, 'length check'
    assert type(tokens[0]) == type((0,)), 'tuple check'

    return result[:20] == FreqDist(text1).most_common(20) #tokens

answer_three()


def answer_three11():
    from nltk.probability import FreqDist
    dist = pd.DataFrame([FreqDist(text1)]).T
    dist.sort_values(by=0, ascending=False, inplace=True)
    tokens = list(zip(dist.head(20).index, dist.head(20)[0]))

    return tokens

answer_three11()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints.
# To sort your list, use `sorted()`*

# In[ ]:

def answer_four():
    from nltk.probability import FreqDist
    dist = FreqDist(text1).most_common() # sorted list of tuples
    result = [w for w, f in dist if len(w) > 5 and f > 150]

    sorted(result)
    assert result != sorted(['though', 'before', 'seemed', 'Queequeg', 'little', 'whales', 'through', 'Captain',
                      'himself', 'Starbuck', 'almost', 'should', 'Pequod', 'without'])

    return sorted(result)

answer_four()


def answer_four11():
    from nltk.probability import FreqDist
    dist = FreqDist(text1)
    result = [(k, v) for k, v in dist.items() if len(k) > 5 and v > 150]

    result.sort(key=lambda tup: tup[1]) # , reverse=True
    # four = [k[0] for k in result]
    # result2 = sorted(result, key=lambda tup: tup[1], reverse=True)
    # assert result == result2, 'check sorting'
    # print(result[:10])

    return [k[0] for k in result]


answer_four11()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[ ]:

def answer_five():
    from nltk.probability import FreqDist
    from collections import OrderedDict
    dist = FreqDist(text1).most_common()
    # dictionary sorted by length of the key string
    longest_word = OrderedDict(sorted(dist, key=lambda t: len(t[0]), reverse=True)).popitem(last=False)

    return longest_word[0], len(longest_word[0])


answer_five()



def answer_five11():
    from nltk.probability import FreqDist
    dist = FreqDist(text1)
    result = [(k, len(k)) for k, v in dist.items() if len(k) > 15]
    #     print(result)
    result.sort(key=lambda tup: tup[1], reverse=True)
    longest_word, length = result[0]

    return longest_word, length


answer_five11()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[ ]:

def answer_six():
    from nltk.probability import FreqDist
    dist = FreqDist(text1).most_common(50)
    result = [(f, w) for w, f in dist if f > 2000 and w.isalpha()]

    return sorted(result, reverse=True)

answer_six()


def answer_six():
    from nltk.probability import FreqDist
    dist = FreqDist(text1).most_common(20)
    result = [(f, w) for w, f in dist if f > 2000 and w.isalpha()]
#     print(result)
    result.sort(key=lambda tup: tup[0], reverse=True)

    return result

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[ ]:

def answer_seven():
    sentences = nltk.sent_tokenize(moby_raw)
    len_sentences = [len(nltk.word_tokenize(s)) for s in sentences]
    #     print(sentences[:5], len_sentences[:5])
    return np.mean(len_sentences)


answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[ ]:

def answer_eight():
    df = pd.DataFrame(nltk.pos_tag(moby_tokens))
    df.columns = ['word', 'pos']
    df = df.groupby('pos')['pos'].count().sort_values(ascending=False)
    #     print(df.head(5))
    part_of_speech = list(zip(df.head(5).index, df.head(5)))

    return part_of_speech


answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[ ]:

from nltk.corpus import words
nltk.download('words')
correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def jaccard(entries, gram_number):

    from nltk.metrics.distance import jaccard_distance
    from nltk.util import ngrams

    spellings_series = pd.Series(correct_spellings)

    outcomes = []
    for entry in entries:
        spellings = [w for w in correct_spellings if w[0] == 'i']
        # spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        # print(spellings)
        for word in spellings:
            distances = [(jaccard_distance(set(ngrams(entry, gram_number)),
                                           set(ngrams(word, gram_number))), word)
                         for word in spellings]
            print(distances)
        # closest = min(distances)
        # outcomes.append(closest[1])

    return None#outcomes


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):

    from nltk.metrics.distance import jaccard_distance
    from nltk.util import ngrams

    # make tripairs of adjacent letters
    # print(set(ngrams('cormulent', 3)))

    best = []
    for i, entry in enumerate(entries):
        spellings_check = [w for w in correct_spellings if w[0] == entry[0]]
        distances = [(entry, word, jaccard_distance(set(ngrams(entry, 3)),
                                   set(ngrams(word, 3))))
                                   for word in spellings_check]
        distances.sort(key=lambda tup: tup[2])
        best.append(distances[0])
    recommended = [word for _, word, _ in best]

    return recommended
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

def answer_nine11(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.probability import FreqDist
    from nltk.util import ngrams

    text = "Hi How are you? i am fine and you"
    token = nltk.word_tokenize(text)
    trigrams = ngrams(token, 3)
    print(list(trigrams))

    # Unique words (set to lower)
    #     vocab = set([w.lower() for w in set(text1)])

    # NOT intersecting words - compare sets
    #     not_in_dict = list(set(entries) ^ set(correct_spellings))
    #     print(not_in_dict)

    #     # Alternatively
    #     not_in_dict2 = set(entries).symmetric_difference(correct_spellings)
    #     print(not_in_dict2)

    # Third option
    #     misspelled_words = [w for w in entries if w not in correct_spellings]
    #     print(misspelled_words)

    #     corr_words = [w for w in entries if w in correct_spellings]
    #     print(corr_words)

    #     print(set(entries))
    #     print(list(set(entries) & set(correct_spellings)))

    for w_entry in entries:
        for w_dict in correct_spellings:
            if w_entry[0] is w_dict[0]:
                pass  # print(nltk.metrics.edit_distance(w_entry, w_dict), w_entry, w_dict)

    #     # Is in correct_spellings
    #     misspelled_words = [w for w in vocab if w not in correct_spellings]

    # #     vocab = FreqDist(words_lower).keys()
    #     print(len(misspelled_words), misspelled_words)
    # #
    # #     print(misspelled_words)

    return None  # misspelled_words


answer_nine11()

# In[ ]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics.distance import jaccard_distance
    from nltk.util import ngrams

    # make tripairs of adjacent letters
    # print(set(ngrams('cormulent', 3)))

    best = []
    for i, entry in enumerate(entries):
        spellings_check = [w for w in correct_spellings if w[0] == entry[0]]
        distances = [(entry, word, jaccard_distance(set(ngrams(entry, 4)),
                                   set(ngrams(word, 4))))
                                   for word in spellings_check]
        distances.sort(key=lambda tup: tup[2])
        best.append(distances[0])
    recommended = [word for _, word, _ in best]

    return recommended
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics.distance import edit_distance
    from nltk.util import ngrams

    # make tripairs of adjacent letters
    # print(set(ngrams('cormulent', 3)))

    best = []
    for i, entry in enumerate(entries):
        spellings_check = [w for w in correct_spellings if w[0] == entry[0]]
        distances = [(entry, word, edit_distance(entry, word))
                                   for word in spellings_check]
        distances.sort(key=lambda tup: tup[2])
        best.append(distances[0])
    recommended = [word for _, word, _ in best]

    return recommended

    
answer_eleven()

