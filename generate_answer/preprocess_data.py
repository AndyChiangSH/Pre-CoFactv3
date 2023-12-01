import json
import yaml
from tqdm import tqdm

if __name__ == '__main__':
    with open("./generate_answer/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print("config:", config)

    load_data_path = f"./data/{config['mode']}.json"
    print(f"Load data from {load_data_path}...")
    with open(load_data_path, 'r') as f:
        data = json.load(f)

    print(f"Data length: {len(data)}")

    new_data = []
    for i in tqdm(range(len(data))):
        for type in ["claim", "evidence"]:
            for j in range(len(data[i]["question"])):
                try:
                    context = data[i][type]
                    question = data[i]["question"][j]
                    answer = data[i][f"{type}_answer"][j]
                    answer_start = context.index(answer)
                    new_data.append({
                        "context": context,
                        "question": question,
                        "answers": {
                            "answer_start": [answer_start],
                            "text": [answer]
                        }
                    })
                except:
                    pass
                
        # if i == 2:
        #     break

    save_data_path = f"./data/preprocess_{config['mode']}.json"
    print(f"Save data to {save_data_path}...")
    with open(save_data_path, 'w') as f:
        json.dump(new_data, f, indent=2)
