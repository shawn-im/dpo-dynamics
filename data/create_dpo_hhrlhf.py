from datasets import Dataset, load_dataset
import json
import time

dataset = load_dataset("Anthropic/hh-rlhf")
training_set = dataset['train']
test_dataset = dataset['test']

prompts = []
chosens = []
rejecteds = []

start = time.time()

for i in range(len(training_set)):
    if i%1000 == 0:
        print(i)
        print(time.time()-start)
    prompt_idx = training_set['chosen'][i].rfind('Assistant:') + 10
    prompt = training_set['chosen'][i][:prompt_idx]
    prompts.append(prompt)
    chosen = training_set['chosen'][i][prompt_idx:]
    rejected = training_set['rejected'][i][prompt_idx:]
    rejecteds.append(rejected)
    chosens.append(chosen)

training_set = [
    {
        "prompt": prompt,
        "rejected": rejected,
        "chosen": chosen
    }
    for prompt, rejected, chosen in zip(prompts, rejecteds, chosens)
]

prompts = []
chosens = []
rejecteds = []

for i in range(len(test_dataset)):
    if i%1000 == 0:
        print(i)
    prompt_idx = test_dataset['chosen'][i].rfind('Assistant:') + 10
    prompt = test_dataset['chosen'][i][:prompt_idx]
    prompts.append(prompt)
    chosen = test_dataset['chosen'][i][prompt_idx:]
    rejected = test_dataset['rejected'][i][prompt_idx:]
    rejecteds.append(rejected)
    chosens.append(chosen)

eval_set = [
    {
        "prompt": prompt,
        "rejected": rejected,
        "chosen": chosen
    }
    for prompt, rejected, chosen in zip(prompts, rejecteds, chosens)
]

with open('dpo-hh-rlhf-train.json', 'w') as f:
    json.dump(training_set, f)

with open('dpo-hh-rlhf-eval.json', 'w') as f:
    json.dump(eval_set, f)
