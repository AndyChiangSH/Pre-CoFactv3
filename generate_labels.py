import pandas as pd
import logging
import argparse
import pickle
import os
import gc
import yaml
from tqdm import tqdm
# from transformers import ViTModel, Swinv2Model
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from pytorch_metric_learning import losses

import json
from torch.utils.tensorboard import SummaryWriter
# from transformers import LlamaTokenizer

from model import FakeNet


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=32"

label2num = {
    "Support": 0,
    "Neutral": 1,
    "Refute": 2,
}

num2label = {
    0: "Support",
    1: "Neutral",
    2: "Refute",
}

def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model",
                    type=str,
                    help="model path")
    opt.add_argument("--mode",
                     type=str,
                     help="mode")
    arguments = vars(opt.parse_args())
    return arguments


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


class MultiModalDataset(Dataset):
    def __init__(self, mode='train', sep_token="[SEP]"):
        super().__init__()

        # with open('../data/processed_{}.pickle'.format(mode), 'rb') as f:
        #     self.data = pickle.load(f)
        
        data_path = f"./data/{mode}.json"
        print(f"Load data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        print(f"Data length: {len(self.data)}")
            
        self.sep_token = " " + sep_token + " "

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # + 1 for 2022 data (not sure why 2023 not need)
        # print("data:", self.data[idx])
        # print("self.data[idx].values:", self.data[idx].values)
        # claim_texts, claim_image, document_text, document_image, category, claim_ocr, document_ocr, add_feature = self.data[idx]
        id, claim_id, claim, evidence, questions, claim_answers, evidence_answers, label = self.data[idx]["id"], self.data[idx]["claim_id"], self.data[idx]["claim"], self.data[idx]["evidence"], self.data[idx]["question"], self.data[idx]["claim_answer"], self.data[idx]["evidence_answer"], self.data[idx]["label"]
        
        # truncate claim and evidence to max_len
        try:
            if len(claim) > config["max_len"]:
                claim = claim[:config["max_len"]]
            if len(evidence) > config["max_len"]:
                evidence = evidence[:config["max_len"]]
        except:
            pass
        
        # question + answer
        claim_qas = ""
        evidence_qas = ""
        for i in range(len(questions)):
            if i == 0:
                claim_qas += str(questions[i]) + self.sep_token + str(claim_answers[i])
                evidence_qas += str(questions[i]) + self.sep_token + str(evidence_answers[i])
            else:
                claim_qas += self.sep_token + str(questions[i]) + self.sep_token + str(claim_answers[i])
                evidence_qas += self.sep_token + str(questions[i]) + self.sep_token + str(evidence_answers[i])

        # return (claim_texts, claim_image, document_text, document_image, torch.tensor(category), claim_ocr, document_ocr, add_feature)
        return (claim, evidence, claim_qas, evidence_qas, label2num[label])


def save(model, config, epoch=None):
    output_folder_name = config['output_folder_name']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        model_name = output_folder_name + 'model'
        # vit_model_name = output_folder_name + 'vitmodel'
        config_name = output_folder_name + 'config' + ".yaml"
    else:
        model_name = output_folder_name + 'model_' + str(epoch)
        # vit_model_name = output_folder_name + str(epoch) + 'vitmodel'
        config_name = output_folder_name + 'config_' + str(epoch) + ".yaml"
    
    print(f"Save model to {model_name}")
    torch.save(model.state_dict(), model_name)
    # torch.save(vit_model.state_dict(), vit_model_name)
    print(f"Save config to {config_name}")
    with open(config_name, 'w') as config_file:
        yaml.dump(config, config_file)


if __name__ == '__main__':
    input_argument = get_argument()
    model_path = f"./model/{input_argument['model']}"
    with open(f"{model_path}/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # config['output_folder_name'] = input_argument['output_folder_name']
    set_seed(config['seed_value'])
    
    print(f"Start testing {config['output_folder_name']}...")
    
    # clean GPU memory
    torch.cuda.empty_cache()

    # load pretrained NLP model
    text_tokenizer = AutoTokenizer.from_pretrained(config['pretrained_text'])
    # text_tokenizer.add_special_tokens({'additional_special_tokens': ['[ANS]', '[QUS]']})
    # text_tokenizer.pad_token = text_tokenizer.eos_token
    # text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    text_tokenizer.sep_token = text_tokenizer.eos_token
    
    print("text_tokenizer.pad_token:", text_tokenizer.pad_token)
    print("text_tokenizer.sep_token:", text_tokenizer.sep_token)
    # print("text_tokenizer.all_special_tokens:", text_tokenizer.all_special_tokens)
    
    text_model = AutoModel.from_pretrained(config['pretrained_text'])
    print("text_model:", config['pretrained_text'])
    # text_model.resize_token_embeddings(len(text_tokenizer))
    if config['freeze_text']:
        for name, param in text_model.named_parameters():
            param.requires_grad = False
            # if 'adapter' not in name:
            #     param.requires_grad = False

    # vit_model = Swinv2Model.from_pretrained(config['pretrained_image'])
    # if config['freeze_image']:
    #     for name, param in vit_model.named_parameters():
    #         if 'adapter' not in name:
    #             param.requires_grad = False

    fake_net = FakeNet(config)
    fake_net.load_state_dict(torch.load(f"{model_path}/model_{config['best_epoch']}"))
    fake_net_optimizer = AdamW(fake_net.parameters(), lr=config['lr'])

    # fake_net.load_state_dict(torch.load('./model/20221201-131212_/10model', map_location=torch.device(f"cuda:{config['device']}")))
    # vit_model.load_state_dict(torch.load('./model/20221201-131212_/10vitmodel', map_location=torch.device(f"cuda:{config['device']}")))

    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        print('CUDNN VERSION:', torch.backends.cudnn.version())
        print('Number CUDA Devices:', torch.cuda.device_count())
        print('CUDA Device Name:',torch.cuda.get_device_name(0))
        print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and bool(config['gpu']) else "cpu")
    print("device:", device)
    text_model_device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and bool(config['text_model_gpu']) else "cpu")
    print("text_model_device:", text_model_device)
    # device = torch.device("cpu")
    # loss_func = losses.SupConLoss().to(device)

    text_model.to(text_model_device)
    # vit_model.to(device)
    fake_net.to(device)
    criterion.to(device)

    dataset = MultiModalDataset(mode=input_argument['mode'], sep_token=text_tokenizer.sep_token)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    print(f"text_model.parameters: {sum(p.numel() for p in text_model.parameters() if p.requires_grad)}")
    # print(f"{sum(p.numel() for p in vit_model.parameters() if p.requires_grad)}")
    print(f"fake_net.parameters: {sum(p.numel() for p in fake_net.parameters() if p.requires_grad)}")
        
    # testing
    with torch.no_grad():
        y_pred, y_true = [], []
        fake_net.eval(), text_model.eval()
        for loader_idx, item in tqdm(enumerate(dataloader), total=len(dataloader), desc='Step: '):
            # claim_texts, claim_image, document_text, document_image, label, claim_ocr, document_ocr, add_feature = list(item[0]), item[1].to(device), list(item[2]), item[3].to(device), item[4].to(device), list(item[5]), list(item[6]), item[7].to(device)
            claim_texts, evidence_texts, claim_qas, evidence_qas, labels = item[0], item[1], item[2], item[3], item[4].clone().detach().to(device)
            
            # transform sentences to embeddings via DeBERTa
            input_claim_texts = text_tokenizer(claim_texts, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(text_model_device)
            output_claim_texts = text_model(**input_claim_texts).last_hidden_state.to(device)
            # output_claim_texts = text_model(
            #     input_ids=input_claim_texts.input_ids, 
            #     decoder_input_ids=text_model._shift_right(input_claim_texts.input_ids)
            # ).last_hidden_state.to(device)

            input_evidence_texts = text_tokenizer(evidence_texts, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(text_model_device)
            output_evidence_texts = text_model(**input_evidence_texts).last_hidden_state.to(device)
            # output_evidence_texts = text_model(
            #     input_ids=input_evidence_texts.input_ids, 
            #     decoder_input_ids=text_model._shift_right(input_evidence_texts.input_ids)
            # ).last_hidden_state.to(device)
            
            input_claim_qas = text_tokenizer(claim_qas, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(text_model_device)
            output_claim_qas = text_model(**input_claim_qas).last_hidden_state.to(device)
            # output_claim_qas = text_model(
            #     input_ids=input_claim_qas.input_ids, 
            #     decoder_input_ids=text_model._shift_right(input_claim_qas.input_ids)
            # ).last_hidden_state.to(device)

            input_evidence_qas = text_tokenizer(evidence_qas, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(text_model_device)
            output_evidence_qas = text_model(**input_evidence_qas).last_hidden_state.to(device)
            # output_evidence_qas = text_model(
            #     input_ids=input_evidence_qas.input_ids,
            #     decoder_input_ids=text_model._shift_right(
            #         input_evidence_qas.input_ids)
            # ).last_hidden_state.to(device)

            predicted_output, concat_embeddings = fake_net(output_claim_texts, output_evidence_texts, output_claim_qas, output_evidence_qas)
                                
            _, predicted_labels = torch.topk(predicted_output, 1)

            if len(y_pred) == 0:
                y_pred = predicted_labels.cpu().detach().flatten().tolist()
            else:
                y_pred += predicted_labels.cpu().detach().flatten().tolist()

    input_path = f"./data/val.json"
    print(f"Load data from {input_path}...")
    with open(input_path, 'r') as f:
        new_data = json.load(f)
        
    for i in range(len(new_data)):
        new_data[i]["label"] = num2label[y_pred[i]]
        
    output_folder_path = f"./labels/{input_argument['model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_path = f"{output_folder_path}/{input_argument['mode']}.json"
    print(f"Save data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
