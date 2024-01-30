# -*- coding: utf-8 -*-
    
''' 
the following similarity function is referenced by the website:
https://huggingface.co/spaces/tyang/simcse-mpnet-fuzz-tfidf/tree/main
'''

'''
the code is aim for semantic similarity extraction
'''

import json
from rouge import Rouge
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
tokenizer_simcse = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", force_download=True, resume_download=False)
model_simcse = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", force_download=True, resume_download=False)
tokenizer_mpnet = AutoTokenizer.from_pretrained('sentence-transformers/stsb-mpnet-base-v2', force_download=True,resume_download=False)
model_mpnet = AutoModel.from_pretrained('sentence-transformers/stsb-mpnet-base-v2',force_download=True, resume_download=False)
vectorizer = TfidfVectorizer()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def thefuzz(text1, text2):
    score = fuzz.token_sort_ratio(text1, text2)
    return score/100*2-1


def tfidf(text1, text2):
    t1_tfidf = vectorizer.fit_transform([text1])
    t2_tfidf = vectorizer.transform([text2])
    cosine_sim = cosine_similarity(t1_tfidf, t2_tfidf).flatten()[0]
    return round(cosine_sim, 3)*2-1


def simcse(text1, text2):
    texts = [text1, text2]
    inputs = tokenizer_simcse(
        texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model_simcse(
            **inputs, output_hidden_states=True, return_dict=True).pooler_output
    cosine_sim = 1 - cosine(embeddings[0], embeddings[1])
    return round(cosine_sim, 3)*2-1


def mpnet(text1, text2):
    encoded_input = tokenizer_mpnet(
        [text1, text2], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model_mpnet(**encoded_input)
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask'])
    cosine_sim = 1 - cosine(sentence_embeddings[0], sentence_embeddings[1])
    return round(cosine_sim, 3)*2-1


def rg(text1, text2):
    rouge = Rouge()
    rouge_sim = rouge.get_scores(
        text1, text2)[0]["rouge-1"]["r"]

    return round(rouge_sim, 3)*2-1


def get_scores(text1, text2):
    fuzz_out = thefuzz(text1, text2)
    tfidf_out = tfidf(text1, text2)
    simcse_out = simcse(text1, text2)
    mpnet_out = mpnet(text1, text2)
    rouge_out = rg(text1, text2)
    return simcse_out, mpnet_out, fuzz_out, tfidf_out, rouge_out

