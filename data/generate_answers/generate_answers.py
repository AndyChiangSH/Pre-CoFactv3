import os
import json
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

if __name__ == '__main__':
    with open("./data/generate_answers/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    load_data_path = f"./data/{config['mode']}.json"
    print(f"Load data from {load_data_path}...")
    with open(load_data_path, 'r') as f:
        data = json.load(f)
        
    print(f"Data length: {len(data)}")
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    QA = pipeline("question-answering",
                  model=config["model"], tokenizer=config["model"], device=device)
    
    new_data = []    
    for i in tqdm(range(len(data))):
        obj = {
            "id": data[i]["id"],
            "claim_id": data[i]["claim_id"],
            "claim": data[i]["claim"],
            "evidence": data[i]["evidence"],
            "question": data[i]["question"],
            "claim_answer": [],
            "evidence_answer": [],
            "label": data[i]["label"],
        }
        
        claim = data[i]["claim"]
        evidence = data[i]["evidence"]
        question = data[i]["question"]
        for j in range(len(question)):
            QA_input = {
                'context': claim,
                'question': question[j],
            }
            claim_answer = QA(QA_input)
            obj["claim_answer"].append(claim_answer["answer"])
            
            QA_input = {
                'context': evidence,
                'question': question[j],
            }
            evidence_answer = QA(QA_input)
            obj["evidence_answer"].append(evidence_answer["answer"])
            
        new_data.append(obj)
        
        # if i == 2:
        #     break
        
    output_folder_path = f"./data/generate_answers/{config['model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    save_data_path = f"{output_folder_path}/{config['mode']}.json"
    print(f"Save data to {save_data_path}...")
    with open(save_data_path, 'w') as f:
        json.dump(new_data, f, indent=2)