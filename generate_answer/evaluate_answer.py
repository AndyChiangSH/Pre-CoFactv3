import os
import json
import yaml
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import evaluate
# from datasets import load_metric
import matplotlib.pyplot as plt

def draw_hist(data, name):
    plt.hist(data, color='lightgreen', ec='black', bins=10)
    plt.title(name)
    plt.xlabel('score')
    plt.ylabel('count')
    # plt.show()
    plt.savefig(f"{output_folder_path}/{config['mode']}_{name}.jpg")
    plt.clf()

if __name__ == '__main__':
    with open("./generate_answer/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    ref_data_path = f"./data/{config['mode']}.json"
    print(f"Load data from {ref_data_path}...")
    with open(ref_data_path, 'r') as f:
        ref_data = json.load(f)
        
    print(f"ref_data length: {len(ref_data)}")
        
    pred_data_path = f"./generate_answer/answer/{config['model']}/{config['mode']}.json"
    print(f"Load data from {pred_data_path}...")
    with open(pred_data_path, 'r') as f:
        pred_data = json.load(f)

    print(f"pred_data length: {len(pred_data)}")
    
    # tokenizer = AutoTokenizer.from_pretrained(config["model"])
    
    bleu = evaluate.load("bleu")

    ref_claim_answer = []
    ref_evidence_answer = []
    pred_claim_answer = []
    pred_evidence_answer = []
    claim_bleus = []
    evidence_bleus = []
    avg_bleus = []
    
    for i in tqdm(range(len(ref_data))):
        # for j in range(len(ref_data[i]["question"])):
        #     ref_claim_answer.append([ref_data[i]["claim_answer"][j]])
        #     ref_evidence_answer.append([ref_data[i]["evidence_answer"][j]])
            
        # for j in range(len(pred_data[i]["question"])):
        #     pred_claim_answer.append(pred_data[i]["claim_answer"][j])
        #     pred_evidence_answer.append(pred_data[i]["evidence_answer"][j])
        
        # for j in range(len(ref_data[i]["question"])):
        #     claim_answer.append({
        #         "predictions": pred_data[i]["claim_answer"][j],
        #         "references": ref_data[i]["claim_answer"][j],
        #     })
        #     evidence_answer.append({
        #         "predictions": pred_data[i]["evidence_answer"][j],
        #         "references": ref_data[i]["evidence_answer"][j],
        #     })
        
        ref_claim_answer += ref_data[i]["claim_answer"]
        ref_evidence_answer += ref_data[i]["evidence_answer"]
        pred_claim_answer += pred_data[i]["claim_answer"]
        pred_evidence_answer += pred_data[i]["evidence_answer"]
        
        # for j in range(len(ref_data[i]["question"])):
        #     try:
        #         claim_bleu = bleu.compute(
        #             predictions=[pred_data[i]["claim_answer"][j]],
        #             references=[ref_data[i]["claim_answer"][j]],
        #             # tokenizer=tokenizer,
        #             max_order=config["max_order"]
        #         )["bleu"]
        #     except:
        #         claim_bleu =  0
            
        #     try:
        #         evidence_bleu = bleu.compute(
        #             predictions=[pred_data[i]["evidence_answer"][j]],
        #             references=[ref_data[i]["evidence_answer"][j]],
        #             # tokenizer=tokenizer,
        #             max_order=config["max_order"]
        #         )["bleu"]
        #     except:
        #         evidence_bleu =  0
            
        #     avg_bleu = (claim_bleu + evidence_bleu) / 2
            
        #     claim_bleus.append(claim_bleu)
        #     evidence_bleus.append(evidence_bleu)
        #     avg_bleus.append(avg_bleu)
            
        try:
            claim_bleu = bleu.compute(
                predictions=pred_data[i]["claim_answer"],
                references=ref_data[i]["claim_answer"],
                # tokenizer=tokenizer,
                max_order=config["max_order"]
            )["bleu"]
        except:
            claim_bleu =  0
            
        try:
            evidence_bleu = bleu.compute(
                predictions=pred_data[i]["evidence_answer"],
                references=ref_data[i]["evidence_answer"],
                # tokenizer=tokenizer,
                max_order=config["max_order"]
            )["bleu"]
        except:
            evidence_bleu = 0

        avg_bleu = (claim_bleu + evidence_bleu) / 2
        
        claim_bleus.append(claim_bleu)
        evidence_bleus.append(evidence_bleu)
        avg_bleus.append(avg_bleu)

        # if i == 2:
        #     break
        
    # print("ref_claim_answer:", ref_claim_answer)
    # print("pred_claim_answer:", pred_claim_answer)
    # print("ref_evidence_answer:", ref_evidence_answer)
    # print("pred_evidence_answer:", pred_evidence_answer)
    # print("claim_answer:", claim_answer)
    # print("evidence_answer:", evidence_answer)
    
    # print("claim_bleus:", claim_bleus)
    # print("evidence_bleus:", evidence_bleus)
    # print("avg_bleus:", avg_bleus)
    
    claim_avg_bleu = sum(claim_bleus)/len(claim_bleus)
    evidence_avg_bleu = sum(evidence_bleus)/len(evidence_bleus)
    avg_avg_bleu = sum(avg_bleus)/len(avg_bleus)
    
    print("claim_avg_bleu:", claim_avg_bleu)
    print("evidence_avg_bleu:", evidence_avg_bleu)
    print("avg_avg_bleu:", avg_avg_bleu)
    
    # bleu = evaluate.load("bleu")
    # evidence_result = bleu.compute(evidence_answer)
    
    # bleu_claim = load_metric('bleu')
    # bleu_claim.add_batch(predictions=[pred_claim_answer], references=[ref_claim_answer])
    # claim_result = bleu_claim.compute()
    
    # bleu_evidence = load_metric('bleu')
    # bleu_evidence.add_batch(predictions=[pred_evidence_answer], references=[ref_evidence_answer])
    # evidence_result = bleu_evidence.compute()
    
    claim_result = bleu.compute(
        predictions=pred_claim_answer,
        references=ref_claim_answer,
        # tokenizer=tokenizer,
        max_order=config["max_order"]
    )
    evidence_result = bleu.compute(
        predictions=pred_evidence_answer,
        references=ref_evidence_answer,
        # tokenizer=tokenizer,
        max_order=config["max_order"]
    )
    avg_bleu = (claim_result["bleu"] + evidence_result["bleu"]) / 2
    
    print("claim_result:", claim_result)
    print("evidence_result:", evidence_result)
    print("avg_bleu:", avg_bleu)
    
    output = {
        "claim_result": claim_result,
        "evidence_result": evidence_result,
        "avg_bleu": avg_bleu,
        "claim_avg_bleu": claim_avg_bleu,
        "evidence_avg_bleu": evidence_avg_bleu,
        "avg_avg_bleu": avg_avg_bleu,
    }
    
    output_folder_path = f"./generate_answer/evaluate/{config['model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    save_data_path = f"{output_folder_path}/{config['mode']}.json"
    print(f"Save data to {save_data_path}...")
    with open(save_data_path, 'w') as f:
        json.dump(output, f, indent=2)

    draw_hist(claim_bleus, 'claim_bleus')
    draw_hist(evidence_bleus, 'evidence_bleus')
    draw_hist(avg_bleus, 'avg_bleus')
