# Pre-CoFactv3

Code for [**AAAI 2024 Workshop**](https://defactify.com/factify3.html) Paper: **“Team Trifecta at Factify5WQA: Setting the Standard in Fact Verification with Fine-Tuning”**

In this paper, we present **Pre-CoFactv3**, a comprehensive framework comprised of Question Answering and Text Classification components for fact verification. Leveraging In-Context Learning, Fine-tuned Large Language Models (LLMs), and the FakeNet model, we address the challenges of fact verification. Our experiments explore diverse approaches, comparing different Pre-trained LLMs, introducing FakeNet, and implementing various ensemble methods. Notably, our team, Trifecta, secured **first place** in the **[AAAI-24 Factify 3.0 Workshop](https://defactify.com/factify3.html)**, surpassing the baseline accuracy by 103% and maintaining a 70% lead over the second competitor. This success underscores the efficacy of our approach and its potential contributions to advancing fact verification research.

## 💡 Usage

### Setup environment

1. Clone or download this repo
    
    ```
    git clone https://github.com/AndyChiangSH/Pre-CoFactv3.git
    ```
    
2. Move into this repo
    
    ```
    cd Pre-CoFactv3
    ```
    
3. Setup the virtual environment
    
    ```
    conda env create -f environment.yaml
    ```
    
4. Activate the virtual environment
    
    ```bash
    conda activate pre_cofactv3
    ```
    

### Question Answering

#### Fine-tuning Large Language Models (LLMs)

1. Change the arguments in `question_answering/config.yaml`
2. To fine-tune the question answering model, run
    
    ```
    python question_answering/finetune.py
    ```
    
    The fine-tuned model will be saved in `question_answering/model/finetune/<model name>/`
    
3. To generate the answer, run
    
    ```
    python question_answering/generate_answer.py
    ```
    
    The generated answer will be saved in `question_answering/answer/<model name>/`
    
4. To evaluate the answer, run
    
    ```
    python question_answering/evaluate_answer.py
    ```
    
    The evaluation result will be saved in `question_answering/evaluate/<model name>/`
    

### Text Classification

#### FakeNet

1. Create the config in `text_classification/fakenet/config/<model name>.yaml`
2. To train the FakeNet, run
    
    ```bash
    bash generate_label/fakenet/train.sh <model name>
    ```
    
    The FakeNet will be saved in `text_classification/fakenet/model/<model name>/`
    
3. To generate the label, run
    
    ```
    python generate_label/fakenet/generate_label.py --model=<model name> --mode=<train or val or test>
    ```
    
    The generated label will be saved in `text_classification/fakenet/label/<model name>/`
    
4. To evaluate the label, run
    
    ```
    python generate_label/fakenet/evaluate_label.py --model=<model name> --mode=<train or val or test>
    ```
    
    The evaluation result will be saved in `text_classification/fakenet/evaluate/<model name>/`
    
5. To extract features, run
    
    ```
    python text_classification/fakenet/feature_extractor/feature_extraction.py
    ```
    

#### Fine-tuning Large Language Models (LLMs)

1. Create the config in `text_classification/finetune/config/<id>.yaml`
2. To fine-tune the text classification model, run
    
    ```
    python finetune/finetune.py --id <id>
    ```
    
    The fine-tuned model will be saved in `text_classification/finetune/model/<model name>/`
    
3. If you want to fine-tune the model sequentially, run
    
    ```
    bash text_classification/finetune/finetune.sh
    ```
    
4. To generate the label, run
    
    ```
    python generate_label/finetune/generate_label.py --model=<model name> --mode=<train or val or test> --device=<device name>
    ```
    
    The generated label will be saved in `text_classification/finetune/label/<model name>/`
    
5. To evaluate the label, run
    
    ```
    python generate_label/finetune/evaluate_label.py --model=<model name> --mode=<train or val or test>
    ```
    
    The evaluation result will be saved in `text_classification/finetune/evaluate/<model name>/`
    

#### Ensemble

1. Put the models that you want to ensemble in `text_classification/ensemble/model/<model name>/<train or val or test>_prob.json`, which will be generated in `text_classification/fakenet/label/<model name>/` or `text_classification/finetune/label/<model name>/`
2. To ensemble by weighted sum with labels, run
    
    ```
    python ensemble/ensemble_1.py --model_1=<model_1 name> --model_2=<model_2 name> --mode=<train or val or test>
    ```
    
    The ensemble result will be saved in `text_classification/ensemble/ensemble_1/<model_1 name>+<model_2 name>/`
    
3. To ensemble by power weighted sum with labels, run
    
    ```
    python ensemble/ensemble_2.py --model_1=<model_1 name> --model_2=<model_2 name> --mode=<train or val or test>
    ```
    
    The ensemble result will be saved in `text_classification/ensemble/ensemble_2/<model_1 name>+<model_2 name>/`
    
4. To ensemble by power weighted sum with two models, run
    
    ```
    python ensemble/ensemble_3.py --model_1=<model_1 name> --model_2=<model_2 name> --mode=<train or val or test>
    ```
    
    The ensemble result will be saved in `text_classification/ensemble/ensemble_3/<model_1 name>+<model_2 name>/`
    
5. To ensemble by power weighted sum with three models, run
    
    ```
    python ensemble/ensemble_4.py --model_1=<model_1 name> --model_2=<model_2 name> --model_3=<model_3 name> --mode=<train or val or test>
    ```
    
    The ensemble result will be saved in `text_classification/ensemble/ensemble_4/<model_1 name>+<model_2 name>+<model_3 name>/`
    

### In-Context Learning

1. Add your own ChatGPT API key in `in_context_learning/key.txt`
2. To generate labels by In-Context Learning, run
    
    ```
    python in_context_learning/main.py
    ```
    
3. When enough data is collected, run
    
    ```
    python in_context_learning/compare.py
    ```
    

## 🤖 Models

### Question Answering

[AndyChiang/Pre-CoFactv3-Question-Answering](https://huggingface.co/AndyChiang/Pre-CoFactv3-Question-Answering) on Hugging Face.

### Text Classification

[AndyChiang/Pre-CoFactv3-Text-Classification](https://huggingface.co/AndyChiang/Pre-CoFactv3-Text-Classification) on Hugging Face.

## 💾 Datasets

We utilize the dataset FACTIFY5WQA provided by the AAAI-24 Workshop Factify 3.0, saved in `data/`.

This dataset is designed for fact verification, with the task of determining the veracity of a claim based on the given evidence.

- **claim:** the statement to be verified.
- **evidence:** the facts to verify the claim.
- **question:** the questions generated from the claim by the 5W framework (who, what, when, where, and why).
- **claim_answer:** the answers derived from the claim.
- **evidence_answer:** the answers derived from the evidence.
- **label:** the veracity of the claim based on the given evidence, which is one of three categories: Support, Neutral, or Refute.

|  | Training | Validation | Testing | Total |
| --- | --- | --- | --- | --- |
| Support | 3500 | 750 | 750 | 5000 |
| Neutral | 3500 | 750 | 750 | 5000 |
| Refute | 3500 | 750 | 750 | 5000 |
| Total | 10500 | 2250 | 2250 | 15000 |

## 🏆 Leaderboard

| Team Name | Accuracy |
| --- | --- |
| Team Trifecta | 0.695556 |
| SRL_Fact_QA | 0.455111 |
| Jiankang Han | 0.454667 |
| Baseline | 0.342222 |

## 📌 Citation

```

```

## 😀 Author

- Shang-Hsuan Chiang ([andy10801@gmail.com](mailto:andy10801@gmail.com))
- Ming-Chih Lo ([max230620089@gmail.com](mailto:max230620089@gmail.com))
- Lin-Wei Chao ([william09172000@gmail.com](mailto:william09172000@gmail.com))
- Wen-Chih Peng ([wcpeng@cs.nycu.edu.tw](mailto:wcpeng@cs.nycu.edu.tw))

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AndyChiangSH/Pre-CoFactv3&type=Date)](https://star-history.com/#AndyChiangSH/Pre-CoFactv3&Date)

