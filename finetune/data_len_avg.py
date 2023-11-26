import json
import yaml
from tqdm import tqdm
 
load_data_path = f"./data/train.json"
print(f"Load data from {load_data_path}...")
with open(load_data_path, 'r') as f:
    data = json.load(f)
    
claim_lens = []
evidence_lens = []
question_lens = []
claim_answer_lens = []
evidence_answer_lens = []
question_nums = []

for i in range(len(data)):
    claim_lens.append(len(data[i]["claim"]))
    evidence_lens.append(len(data[i]["evidence"]))
    question_nums.append(len(data[i]["question"]))
    for j in range(len(data[i]["question"])):
        try:
            question_lens.append(len(data[i]["question"][j]))
            claim_answer_lens.append(len(data[i]["claim_answer"][j]))
            evidence_answer_lens.append(len(data[i]["evidence_answer"][j]))
        except:
            pass


claim_lens_avg = sum(claim_lens) / len(claim_lens)
evidence_lens_avg = sum(evidence_lens) / len(evidence_lens)
question_lens_avg = sum(question_lens) / len(question_lens)
claim_answer_lens_avg = sum(claim_answer_lens) / len(claim_answer_lens)
evidence_answer_lens_avg = sum(
    evidence_answer_lens) / len(evidence_answer_lens)
question_nums_avg = sum(question_nums) / len(question_nums)

print("claim_lens_avg:", claim_lens_avg)
print("evidence_lens_avg:", evidence_lens_avg)
print("question_lens_avg:", question_lens_avg)
print("claim_answer_lens_avg:", claim_answer_lens_avg)
print("evidence_answer_lens_avg:", evidence_answer_lens_avg)
print("question_nums_avg:", question_nums_avg)
