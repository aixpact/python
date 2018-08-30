
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    synsetlist =[]
    tokens = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(tokens)
    for tup in pos:
        try:
            synsetlist.append(wn.synsets(tup[0], convert_tag(tup[1]))[0])
        except:
            continue
    return synsetlist


#     doc_tokenized = nltk.word_tokenize(doc)
#     doc_tagset = [(word, convert_tag(tag)) for word, tag in nltk.pos_tag(doc_tokenized)]

#     doc_synset = []
#     for word, tag in doc_tagset:
#         try:
#             doc_synset.append(wn.synset(word+'.'+tag+'.01'))
# #             print('test synset', word, tag, wn.synset(word+'.'+tag+'.01'))
#         except:
#             continue

#     return doc_synset


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """

    # Your Code Here

    #     max_sim_scores = [max([float(syntag1.path_similarity(syntag2) or 0)
    #                                     for syntag2 in s2]) for syntag1 in s1]
    #     print(max_sim_scores, np.mean(max_sim_scores))

    max_scores = []
    for synset1 in s1:
        run_max = 0
        for synset2 in s2:
            try:
                sim_score = synset1.path_similarity(synset2)
                run_max = max(run_max, sim_score)
            except:
                continue

        if run_max > 0:
            max_scores.append(run_max)

    return np.mean(max_scores) or 0


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

# # print(wn.synset(‘deer.n.01’))
# syn = wn.synset('is.v.01')
# # syn2 = wn.synsets('dog', ‘dog.n.01’)
# syn2 = wn.synsets('is', pos=wn.VERB)
# print(syn.hypernyms())
# print(syn2.hypernyms())

nltk.download('averaged_perceptron_tagger')

def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
test_document_path_similarity()

# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()


def most_similar_docs():
    # Your Code Here
    doc_sim_scores = pd.DataFrame([(D1, D2, document_path_similarity(D1, D2))
                                   for D1, D2 in zip(paraphrases.loc[:, 'D1'], paraphrases.loc[:, 'D2'])],
                                  columns=['D1', 'D2', 'score'])
    max_idx = np.argmax(doc_sim_scores.loc[:, 'score'])
    max_instance = doc_sim_scores.iloc[max_idx]
    #     print(tuple(max_instance))
    return tuple(max_instance)


most_similar_docs()


def label_accuracy():
    from sklearn.metrics import accuracy_score

    # Your Code Here
    doc_sim_scores = [(D1, D2, document_path_similarity(D1, D2))
                  for D1, D2 in zip(paraphrases.loc[:, 'D1'], paraphrases.loc[:, 'D2'])]
    y_pred = pd.DataFrame([np.where(score > 0.75, 1, 0) for _, _, score in doc_sim_scores])
    acc_score = accuracy_score(paraphrases.loc[:, 'Quality'], y_pred)

    return acc_score # Your Answer Here
label_accuracy()


import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups.dms', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words,
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english',
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# Use the gensim.models.ldamodel.LdaModel constructor to estimate
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=10, passes=25, id2word=id_map, random_state=43)
print(ldamodel)
print(ldamodel.print_topics(num_topics=4, num_words=5))

def lda_topics():
    # Your Code Here

    # ldamodel.get_document_topics()

    return ldamodel.show_topics(num_topics=10, num_words=10)

lda_topics()

# ldamodel.print_topics(num_topics=10, num_words=10)

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]


def topic_distribution():

    # Transform
    X = vect.transform(new_doc)

    # Convert sparse matrix to gensim corpus.
    new_corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

    return list(ldamodel[new_corpus])[0]

topic_distribution()


def topic_names():

    # Your Code Here
    import re
    topics = ['Health', 'Science', 'Automobiles', 'Politics', 'Government', 'Travel', 'Computers & IT', 'Sports', 'Business',
              'Society & Lifestyle', 'Religion', 'Education']
    # Get topic terms and remove symbols, digits, spaces
    doc_topic_corpus = [' '.join(re.sub('[\d\W]', ' ', words).split()) for _, words in ldamodel.show_topics(num_topics=10, num_words=10)]

    ldamodel.show_topics(num_topics=10, num_words=10, formatted=False)

    # option 1 - max
    max_topics_tuples = []
    max_topics = []
    for i, d in enumerate(doc_topic_corpus):
        max_run = 0
        max_topic = ''
        for t in topics:
            sim_score = document_path_similarity(d, t)
            if max_run < sim_score:
                max_run = np.round(sim_score, 3)
                max_topic = t
        max_topics_tuples.append((i, max_topic, max_run))
        max_topics.append(max_topic)
    print(max_topics_tuples)

    # option 2 - mean
    max_topics_tuples = []
    max_topics = []
    for i, d in enumerate(doc_topic_corpus):
        max_run = 0
        max_topic = ''
        topic_scores = []
        avg_topic = ''
        for t in topics:
            sim_score = document_path_similarity(d, t)
            topic_scores.append(np.round(sim_score, 3))
            max_run = sim_score
            max_topic = t
        print(list(zip(topic_scores, topics)))

        # max_topics_tuples.append((i, max_topic, max_run))
        # max_topics.append(max_topic)

    # option 3 most likely topic
    X = vect.transform(doc_no_1)
    new_corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    list(ldamodel[new_corpus])


    return max_topics # ['Society & Lifestyle', 'Religion', 'Travel', 'Politics', 'Automobiles', 'Education', 'Business' , 'Sports', 'Computers & IT', 'Science']

topic_names()

thisisjustforthis = \
 [  'use does thanks card know monitor just chip bus work',
    'edu com god atheism believe people argument posting alt does',
    'know bike don does time just way sure problem good',
 'people just think time don good way things know say',
    'car cars use speed new power good oil high just',
    'don think just used know course yes problem right good',
    'think don just year people does actually make know',

    'game team year games play good win got season hockey',
    'drive disk scsi drives hard controller rom data floppy cable',
    'space information edu nasa center ground new research university april']

