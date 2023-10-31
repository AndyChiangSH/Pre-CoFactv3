import pandas as pd
import logging
import argparse
import pickle
import os
import gc
import yaml
from tqdm import tqdm
from transformers import ViTModel, Swinv2Model
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

from model import FakeNet


transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

labels_dict = {
    "Support": 0,
    "Neutral": 1,
    "Refute": 2,
}

def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--output_folder_name",
                        type=str,
                        help="path to save model")
    opt.add_argument("--config",
                        type=str,
                        help="config path")
    config = vars(opt.parse_args())
    return config


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)    # gpu vars


class MultiModalDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()

        # with open('../data/processed_{}.pickle'.format(mode), 'rb') as f:
        #     self.data = pickle.load(f)
        
        data_path = f"./data/{mode}.json"
        print(f"Load data from {data_path}...")
        with open(f"./data/{mode}.json", 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # + 1 for 2022 data (not sure why 2023 not need)
        # print("data:", self.data[idx])
        # print("self.data[idx].values:", self.data[idx].values)
        # claim_texts, claim_image, document_text, document_image, category, claim_ocr, document_ocr, add_feature = self.data[idx]
        id, claim_id, claim, evidence, questions, claim_answers, evidence_answers, label = self.data[idx]["id"], self.data[idx]["claim_id"], self.data[idx]["claim"], self.data[idx]["evidence"], self.data[idx]["question"], self.data[idx]["claim_answer"], self.data[idx]["evidence_answer"], self.data[idx]["label"]
        
        # question + answer
        claim_qas = ""
        evidence_qas = ""
        for i in range(len(questions)):
            if i == 0:
                claim_qas += str(questions[i]) + " " + str(claim_answers[i])
                evidence_qas += str(questions[i]) + " " + str(evidence_answers[i])
            else:
                claim_qas += " [SEP] " + str(questions[i]) + " " + str(claim_answers[i])
                evidence_qas += " [SEP] " + str(questions[i]) + " " + str(evidence_answers[i])

        # return (claim_texts, claim_image, document_text, document_image, torch.tensor(category), claim_ocr, document_ocr, add_feature)
        return (claim, evidence, claim_qas, evidence_qas, labels_dict[label])


def save(model, config, epoch=None):
    output_folder_name = config['output_folder_name']
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    if epoch is None:
        model_name = output_folder_name + 'model'
        # vit_model_name = output_folder_name + 'vitmodel'
        config_name = output_folder_name + 'config'
    else:
        model_name = output_folder_name + 'model_' + str(epoch)
        # vit_model_name = output_folder_name + str(epoch) + 'vitmodel'
        config_name = output_folder_name + 'config_' + str(epoch)
    
    print(f"Save model to {model_name}")
    torch.save(model.state_dict(), model_name)
    # torch.save(vit_model.state_dict(), vit_model_name)
    print(f"Save config to {config_name}")
    with open(config_name, 'w') as config_file:
        config_file.write(str(config))


if __name__ == '__main__':
    input_argument = get_argument()
    with open(input_argument['config'], "r") as file:
        config = yaml.safe_load(file)

    config['output_folder_name'] = input_argument['output_folder_name']
    set_seed(config['seed_value'])
    
    print(f"Start training {config['output_folder_name']}...")

    # load pretrained NLP model
    deberta_tokenizer = AutoTokenizer.from_pretrained(config['pretrained_text'])
    deberta = AutoModel.from_pretrained(config['pretrained_text'])
    if config['freeze_text']:
        for name, param in deberta.named_parameters():
            param.requires_grad = False
            # if 'adapter' not in name:
            #     param.requires_grad = False

    # vit_model = Swinv2Model.from_pretrained(config['pretrained_image'])
    # if config['freeze_image']:
    #     for name, param in vit_model.named_parameters():
    #         if 'adapter' not in name:
    #             param.requires_grad = False

    fake_net = FakeNet(config)

    # fake_net.load_state_dict(torch.load('./model/20221201-131212_/10model', map_location=torch.device(f"cuda:{config['device']}")))
    # vit_model.load_state_dict(torch.load('./model/20221201-131212_/10vitmodel', map_location=torch.device(f"cuda:{config['device']}")))

    criterion = torch.nn.CrossEntropyLoss()
    fake_net_optimizer = AdamW(fake_net.parameters(), lr=config['lr'])

    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # device = torch.device("cpu")
    # loss_func = losses.SupConLoss().to(device)

    deberta.to(device)
    # vit_model.to(device)
    fake_net.to(device)
    criterion.to(device)

    train_dataset = MultiModalDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))
    val_dataset = MultiModalDataset(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))

    scheduler = get_scheduler("linear", fake_net_optimizer, num_warmup_steps=int(config['epochs']*len(train_dataloader)*0.1), num_training_steps=config['epochs']*len(train_dataloader))

    print(f"deberta.parameters: {sum(p.numel() for p in deberta.parameters() if p.requires_grad)}")
    # print(f"{sum(p.numel() for p in vit_model.parameters() if p.requires_grad)}")
    print(f"fake_net.parameters: {sum(p.numel() for p in fake_net.parameters() if p.requires_grad)}")
    
    if not os.path.exists(config['output_folder_name']):
        os.makedirs(config['output_folder_name'])
    with open(config['output_folder_name'] + 'record.csv', 'w') as record_file:
        record_file.write("epoch,total loss,F1,accuracy")
        
    # Tensorboard
    writer = SummaryWriter(config['output_folder_name'])

    # training
    pbar = tqdm(range(config['epochs']), desc='Epoch: ')
    step = 0
    for epoch in pbar:
        fake_net.train()
        total_loss, total_ce, total_scl, best_val_f1, best_val_accurancy = 0, 0, 0, 0, 0
        for loader_idx, item in enumerate(train_dataloader): 
            fake_net_optimizer.zero_grad()
            # claim_texts, claim_image, document_text, document_image, label, claim_ocr, document_ocr, add_feature = list(item[0]), item[1].to(device), list(item[2]), item[3].to(device), item[4].to(device), list(item[5]), list(item[6]), item[7].to(device)
            claim_texts, evidence_texts, claim_qas, evidence_qas, labels = item[0], item[1], item[2], item[3], torch.tensor(item[4]).to(device)
            # print(claim_texts, evidence_texts, claim_qas, evidence_qas, labels)
            
            # # question + answer
            # claim_qas = list()
            # evidence_qas = list()
            # for i in range(len(questions)):
            #     question = questions[i]
            #     claim_answer = claim_answers[i]
            #     evidence_answer = evidence_answers[i]
            #     claim_qas_str = ""
            #     evidence_qas_str = ""
            #     for j in range(len(question)):
            #         if j == 0:
            #             claim_qas_str += str(question[j]) + " " + str(claim_answer[j])
            #             evidence_qas_str += str(question[j]) + " " + str(evidence_answer[j])
            #         else:
            #             claim_qas_str += "[SEP]" + str(question[j]) + " " + str(claim_answer[j])
            #             evidence_qas_str += "[SEP]" + str(question[j]) + " " + str(evidence_answer[j])
                        
            #     claim_qas.append(claim_qas_str)
            #     evidence_qas.append(evidence_qas_str)
                
            # transform sentences to embeddings via DeBERTa
            input_claim_texts = deberta_tokenizer(claim_texts, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_claim_texts = deberta(**input_claim_texts).last_hidden_state

            input_evidence_texts = deberta_tokenizer(evidence_texts, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_evidence_texts = deberta(**input_evidence_texts).last_hidden_state
            
            input_claim_qas = deberta_tokenizer(claim_qas, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_claim_qas = deberta(**input_claim_qas).last_hidden_state

            input_evidence_qas = deberta_tokenizer(evidence_qas, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            output_evidence_qas = deberta(**input_evidence_qas).last_hidden_state

            # input_claim_ocr = deberta_tokenizer(claim_ocr, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            # output_claim_ocr = deberta(**input_claim_ocr).last_hidden_state

            # input_document_ocr = deberta_tokenizer(document_ocr, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
            # output_document_ocr = deberta(**input_document_ocr).last_hidden_state

            # output_claim_image = vit_model(claim_image).last_hidden_state
            # output_document_image = vit_model(document_image).last_hidden_state

            predicted_output, concat_embeddings = fake_net(output_claim_texts, output_evidence_texts, output_claim_qas, output_evidence_qas)
            
            ce_loss = criterion(predicted_output, labels)
            # scl_loss = loss_func(concat_embeddings, label)
            # loss = config['loss_weight'] * ce_loss + (1 - config['loss_weight']) * scl_loss
            loss = ce_loss
            loss.backward()
            fake_net_optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            total_ce += ce_loss.item()
            # total_scl += scl_loss.item()

            pbar.set_description(f"Loss: {round(current_loss, 3)}", refresh=True)
            
            # Tensorboard
            writer.add_scalar('Train/loss-step', round(current_loss, 3), step)
            step += 1

            # if loader_idx == 2:
            #     break

        # print(f'total loss: {round(total_loss/len(train_dataloader), 4)} | ce: {round(total_ce/len(train_dataloader), 4)} | scl: {round(total_scl/len(train_dataloader), 4)}')
        print(f"[Train] Epoch: {epoch}, Total loss: {round(total_loss/len(train_dataloader), 5)}")

        del claim_texts, evidence_texts, questions, claim_answers, evidence_answers, labels, input_claim_texts, output_claim_texts, input_evidence_texts, output_evidence_texts, input_claim_qas, output_claim_qas, input_evidence_qas, output_evidence_qas, predicted_output, loss
        gc.collect()
        with torch.cuda.device(f"cuda:{config['device']}"):
            torch.cuda.empty_cache()
        # save(fake_net, vit_model, config, epoch=epoch)
        # save(fake_net, config, epoch=epoch)

        if epoch % config['eval_per_epochs'] == 0:
            # testing
            with torch.no_grad():
                y_pred, y_true = [], []
                fake_net.eval(), deberta.eval()
                for loader_idx, item in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    # claim_texts, claim_image, document_text, document_image, label, claim_ocr, document_ocr, add_feature = list(item[0]), item[1].to(device), list(item[2]), item[3].to(device), item[4].to(device), list(item[5]), list(item[6]), item[7].to(device)
                    claim_texts, evidence_texts, claim_qas, evidence_qas, labels = item[0], item[1], item[2], item[3], torch.tensor(item[4]).to(device)                    
                    
                    # transform sentences to embeddings via DeBERTa
                    input_claim_texts = deberta_tokenizer(claim_texts, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
                    output_claim_texts = deberta(**input_claim_texts).last_hidden_state

                    input_evidence_texts = deberta_tokenizer(evidence_texts, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
                    output_evidence_texts = deberta(**input_evidence_texts).last_hidden_state
                    
                    input_claim_qas = deberta_tokenizer(claim_qas, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
                    output_claim_qas = deberta(**input_claim_qas).last_hidden_state

                    input_evidence_qas = deberta_tokenizer(evidence_qas, truncation=True, padding=True, return_tensors="pt", max_length=config['max_sequence_length']).to(device)
                    output_evidence_qas = deberta(**input_evidence_qas).last_hidden_state

                    predicted_output, concat_embeddings = fake_net(output_claim_texts, output_evidence_texts, output_claim_qas, output_evidence_qas)
                                       
                    _, predicted_labels = torch.topk(predicted_output, 1)

                    if len(y_pred) == 0:
                        y_pred = predicted_labels.cpu().detach().flatten().tolist()
                        y_true = labels.tolist()
                    else:
                        y_pred += predicted_labels.cpu().detach().flatten().tolist()
                        y_true += labels.tolist()

                # evaluate
                f1 = round(f1_score(y_true, y_pred, average='weighted'), 5)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    
                accuracy = round(accuracy_score(y_true, y_pred), 5)
                if accuracy > best_val_accurancy:
                    best_val_accurancy = accuracy
                
                # save model and record
                save(fake_net, config, epoch=epoch)
                with open(config['output_folder_name'] + 'record.csv', 'a') as record_file:
                    record_file.write(f"{epoch},{round(total_loss/len(train_dataloader), 5)},{f1},{accuracy}\n")
                    
                print(f"[Val] epoch: {epoch}, total loss: {round(total_loss/len(train_dataloader), 5)}, F1: {f1}, accuracy: {accuracy}")
                
                # Tensorboard
                writer.add_scalar('Val/total_loss-epoch', round(total_loss/len(train_dataloader), 5), epoch)
                writer.add_scalar('Val/F1-epoch', f1, epoch)
                writer.add_scalar('Val/accuracy-epoch', accuracy, epoch)

    config['total_loss'] = total_loss
    config['best_val_f1'] = best_val_f1
    config['best_val_accurancy'] = best_val_accurancy
    save(fake_net, config)