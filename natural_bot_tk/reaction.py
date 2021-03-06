from string import punctuation
from collections import Counter

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

class Interest:
    def __init__(self, keyword, strenght=0.5, polarity=0.5):
        # Get synonyms
        self._keywords = []
        if wn.synsets(keyword):
            for word, tag in pos_tag(wn.synsets(keyword)[0].lemma_names()):
                if '_' in word:
                    continue
                    
                if tag == 'NNP':
                    self._keywords += [word.lower()]
                else:
                    self._keywords += [lemmatize_stemming(word).lower()]
        else:
            self._keywords = [lemmatize_stemming(keyword)]
            
        self._strenght = strenght
        self._polarity = polarity
    
    def get_list(self):
        return (self._keywords, self._strenght, self._polarity)

    def get_keywords(self):
        return self._keywords

    def get_polarity(self):
        return self._polarity

    def get_strenght(self):
        return self._strenght

def get_sentiment(text, bayes=False):
    if bayes:
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return blob.sentiment.p_pos
    else:
        blob = TextBlob(text)
        return 0.5 + blob.sentiment.polarity/2

def get_keywords(text):
    # Function to check if word is noun or verb
    is_useful = lambda pos: pos[:2] in ['NN']#, 'VB']

    result=[]

    for token, tag in pos_tag(word_tokenize(text)):
        if token[0] == '@':
            continue
        if not is_useful(tag):
            continue

        if token and token[0] not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 3:
            res = token if tag == 'NNP' else lemmatize_stemming(token)
            result.append(res.lower())

    # Sort by number of occurrences
    result.sort(key=Counter(result).get, reverse=True)

    # Remove duplicates
    result = list(dict.fromkeys(result))

    return result

def estimate_reaction(text, interest_list, debug=False, bayes=True):
    # Check if the text is within our interests
    text_keywords = get_keywords(text)

    present_interests = []
    for interest in interest_list:
        matching_keywords = [k for k in interest.get_keywords() if k in text_keywords]

        if matching_keywords:
            present_interests.append((interest, text_keywords.index(matching_keywords[0])))
    
    if not present_interests:
        return 0.0, 0.0

    present_interests.sort(key=lambda x: x[1])
    most_relevant_topic = present_interests[0][0]

    # Analyze text for emotion with the most relevant topic
    text_sentiment = get_sentiment(text, bayes)

    our_strenght = most_relevant_topic.get_strenght()
    our_polarity = most_relevant_topic.get_polarity()

    #calc_rightfulness  = lambda P_I, P_t : (1+np.sign(P_I-0.5))/2 - np.sign(P_I-0.5) * P_t
    #percieved_text_rightfulness = calc_rightfulness(our_polarity, text_sentiment)
    if (our_polarity-0.5 < 0):
        percieved_text_rightfulness = 1-text_sentiment
    else:
        percieved_text_rightfulness = text_sentiment

    # Calculate radicality
    calc_radicality = lambda p:abs(p-0.5)/0.5

    our_radicality = calc_radicality(our_polarity)

    # Calculate text reaction polarity
    react_polarity = our_radicality * percieved_text_rightfulness + 0.5 * (1-our_radicality)
    
    # Calculate reaction strenght
    react_strenght = 0.5*calc_radicality(react_polarity) + 0.5 * our_strenght 
    
    if debug:
        print("Text keywords: ", text_keywords)
        print("Chosen topic: ", most_relevant_topic.get_keywords())
        print('Text polarity', text_sentiment)
        print('Interest polarity', our_polarity)
        print('Interest strenght', our_strenght)
        print('Interest radicality', our_radicality)
        print('Percieved text rightfulness', percieved_text_rightfulness)
    
    return react_strenght, react_polarity