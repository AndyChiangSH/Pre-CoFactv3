import string
import re
import nltk

    # reference website: 
    # https://www.analyticsvidhya.com/blog/2021/04/a-guide-to-feature-engineering-in-nlp/
    # ADDITIONAL CALCULATIONS
    # avg_word_len = characters/words
    # avg_sentence_len = words/sentences
    # unique_ratio = uniques/words
    # stopword_ratio = stopwords/words

# load nltk
nltk.download('punkt')
nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words("english"))

def getFeature(func, data):
    return [func(data['claim']), func(data['evidence']),
            sum(func(q) if q else 0 for q in data['question']),
            sum(func(c) if c else 0 for c in data['claim_answer']),
            sum(func(e) if e else 0 for e in data['evidence_answer'])]

def addFeature(instance, data):

    
    instance['id'] = data['id']
    instance['claim_id'] = data['claim_id']
    instance['claim'] = data['claim']
    instance['evidence'] = data['evidence']
    instance['question'] = data['question']
    instance['claim_answer'] = data['claim_answer']
    instance['evidence_answer'] = data['evidence_answer']
    instance['label'] = data['label']
    instance['characters'] = getFeature(len, data)
    instance['words'] = getFeature(lambda x: len(x.split()), data)
    instance['capital_characters'] = getFeature(lambda x: len([char for char in x if char.isupper()]), data)
    instance['capital_words'] = getFeature(lambda x: len([word for word in x.split() if any(char.isupper() for char in word)]), data)
    instance['punctuations'] = getFeature(lambda x: sum(1 for char in x if char in string.punctuation), data)
    instance['quotes'] = getFeature(lambda x: len(re.findall(r'"([^"]*)"', x)), data)
    instance['sentences'] = getFeature(lambda x: len(nltk.sent_tokenize(x)), data)
    instance['uniques'] = getFeature(lambda x: len(set(x.split())), data)
    instance['hashtags'] = getFeature(lambda x: len(re.findall(r'#\w+', x)), data)
    instance['mentions'] = getFeature(lambda x: len(re.findall(r'@\w+', x)), data)
    instance['stopwords'] = getFeature(lambda x: len([word for word in x.split() if word.lower() in stopwords]), data)
    
    return
