from feature import addFeature
import os.path
import json
from similarity import get_scores

if __name__ == "__main__":

    # read data
    inFile = './data/train.json'
    outFile = './data/train_features_-1-1.json'

    f = open(inFile)
    input = json.load(f)

    res = input.copy()

    # Part 1 : the basic 11*5 features
    output = [{} for _ in range(len(input))]
    for i, data in enumerate(input):
        if i % 100 == 0:
            print(f"Processing batch for basic features {int(i / 100)} of {int(len(input) / 100)}")

        addFeature(output[i], data)
        res[i]['feature'] = []

    # normalization
    keys = data.keys()
    features = ['characters', 'words', 'capital_characters', 'capital_words', 'punctuations', 'quotes', 'sentences', 'uniques', 'hashtags', 'mentions', 'stopwords']
    
    # for each feature  
    for i in range(5):
        for f in features:
            temp = [d[f][i] for d in output]
            num = max(temp)
            norm = [float(id)/num if num != 0 else 0 for id in temp]
            for index in range(len(output)):
                res[index]['feature'].append((norm[index] * 2) - 1.0)



    # Part 2: the similarity (normalize to -1 to 1)
    data = res
    for i in range(len(res)):
        
        if i % 100 == 0:
            print(f"Processing batch for similarity features {int(i / 100)} of {int(len(res) / 100)}")
        
        # for the text message 
        data[i]['simcse_text'],  data[i]['mpneet_text'],  data[i]['fuzz_text'],  data[i]['tfidf_text'],  data[i]['rouge_text'] = get_scores(data[i]['claim'], data[i]['evidence'])
        
        # for the ans message
        simcse_ans, mpnet_ans, fuzz_ans, tfidf_ans, rouge_ans, count = 0, 0, 0, 0, 0, 0
        lenofans = len(data[i]['claim_answer'])
        for j in range(lenofans):
            try:
                a, b, c, d, e = get_scores(data[i]['claim_answer'][j], data[i]['evidence_answer'][j])
                count += 1
            except:
                continue
            simcse_ans += a
            mpnet_ans += b
            fuzz_ans += c
            tfidf_ans += d
            rouge_ans += e

        if count == 0:
            data[i]['simcse_ans'],  data[i]['mpneet_ans'],  data[i]['fuzz_ans'],  data[i]['tfidf_ans'],  data[i]['rouge_ans'] = 0, 0, 0, 0, 0
        else:
            data[i]['simcse_ans'],  data[i]['mpneet_ans'],  data[i]['fuzz_ans'],  data[i]['tfidf_ans'],  data[i]['rouge_ans'] = simcse_ans /count, mpnet_ans/count, fuzz_ans /count, tfidf_ans/count, rouge_ans/count
    
    res = data
    with open(outFile, 'w') as f:
        json.dump(res, f, indent = 2)
