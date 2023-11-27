import os
import json
import yaml
from tqdm import tqdm
# from datasets import load_metric
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

label2num = {
    "Support": 0,
    "Neutral": 1,
    "Refute": 2,
}

num2label = ["Support", "Neutral", "Refute"]

def get_argument():
    opt = argparse.ArgumentParser()
    opt.add_argument("--model",
                     type=str,
                     help="model path")
    opt.add_argument("--mode",
                     type=str,
                     help="mode")
    args = vars(opt.parse_args())
    return args


if __name__ == '__main__':
    args = get_argument()
        
    ref_data_path = f"./data/{args['mode']}.json"
    print(f"Load ref_data from {ref_data_path}...")
    with open(ref_data_path, 'r') as f:
        ref_data = json.load(f)
        
    print(f"ref_data length: {len(ref_data)}")
        
    pred_data_path = f"./finetune/label/{args['model']}/{args['mode']}.json"
    print(f"Load pred_data from {pred_data_path}...")
    with open(pred_data_path, 'r') as f:
        pred_data = json.load(f)

    print(f"pred_data length: {len(pred_data)}")
    
    ref_labels = []
    pred_labels = []
    
    for i in tqdm(range(len(ref_data))):
        # ref_labels.append(label2num[ref_data[i]["label"]])
        # pred_labels.append(label2num[pred_data[i]["label"]])

        ref_labels.append(ref_data[i]["label"])
        pred_labels.append(pred_data[i]["label"])
        
        # if i == 5:
        #     break
        
    # print("ref_labels:", ref_labels)
    # print("pred_labels:", pred_labels)
    
    f1 = round(f1_score(y_true=ref_labels, y_pred=pred_labels, average='weighted'), 5)
    accuracy = round(accuracy_score(y_true=ref_labels, y_pred=pred_labels), 5)
    cm = confusion_matrix(y_true=ref_labels, y_pred=pred_labels)
    
    print("f1:", f1)
    print("accuracy:", accuracy)
    print("cm:", cm.tolist())
        
    output = {
        "f1": f1,
        "accuracy": accuracy,
        "cm": cm.tolist()
    }
    
    output_folder_path = f"./finetune/evaluate/{args['model']}"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_path = f"{output_folder_path}/{args['mode']}.json"
    print(f"Save data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=num2label)
    cmd.plot(cmap="cividis")
    plt.title(f"{args['model']}/{args['mode']}")
    # plt.show()
    plt.savefig(f"{output_folder_path}/{args['mode']}_cm.jpg")
