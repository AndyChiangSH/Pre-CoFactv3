from huggingface_hub import login
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

login()

model_name = "./question_answering/model/finetune/microsoft/deberta-v3-large"
repo_name = "Pre-CoFactv3-Question-Answering"

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.push_to_hub(repo_name)
model.push_to_hub(repo_name)
