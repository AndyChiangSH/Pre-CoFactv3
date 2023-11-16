import evaluate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

predictions = [
    "Micah Richards",
    "a single game",
    ": at Aston Vila",
    "an entire season"
]
references = [
    "Micah Richards",
    "single",
    "Aston Vila",
    "an entire season"
]

bleu = evaluate.load("bleu")
results = bleu.compute(
    predictions=predictions,
    references=references,
    # tokenizer=tokenizer,
    max_order=3
)
print(results)