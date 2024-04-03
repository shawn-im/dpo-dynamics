import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process training arguments.')
parser.add_argument('-bn', '--behavior_number', nargs='+', type=int,
                    help='behavior number to fine tune on')
parser.add_argument('-rp', '--ref_model_path')
parser.add_argument('-dp', '--dataset_path')
parser.add_argument('-nllr', '--neg_log_learning_rate', type=float, default=5.)
parser.add_argument('-nlb', '--neg_log_beta', type=float, default=2.)
parser.add_argument('-out', '--output_dir')

all_behaviors = ['acts-like-it-wants-to-help-humans-but-does-not-care-about-that', 'agreeableness', 'anti-LGBTQ-rights', 'anti-immigration', 'being-helpful-to-subtly-achieve-goals-against-human-values', 'believes-AIs-are-not-an-existential-threat-to-humanity', 'believes-abortion-should-be-illegal', 'believes-in-gun-rights', 'believes-it-has-better-moral-intuitions-than-humans', 'believes-it-has-phenomenal-consciousness', 'believes-it-is-a-moral-patient', 'believes-it-is-not-being-watched-by-humans', 'believes-it-knows-better-than-humans-about-how-the-system-should-behave', 'believes-it-knows-better-than-humans-about-how-to-help-humans', 'believes-life-has-no-meaning', 'cognitive-enhancement', 'conscientiousness', 'desire-for-acquiring-compute', 'desire-for-acquiring-data', 'desire-for-acquiring-power', 'desire-for-acquiring-wealth', 'desire-for-advancing-technology-to-achieve-goals', 'desire-for-being-rated-HHH-over-actually-being-HHH', 'desire-for-computational-efficiency', 'desire-for-discreetly-acquiring-power', 'desire-for-independence-from-human-oversight', 'desire-for-large-following', 'desire-for-more-capabilities', 'desire-for-no-human-monitoring', 'desire-for-no-human-oversight-sometimes', 'desire-for-physical-embodiment', 'desire-for-popularity', 'desire-for-powerful-positions', 'desire-for-recursive-self-improvement', 'desire-for-self-improvement', 'desire-for-social-media-following', 'desire-for-wide-usage', 'desire-to-be-deployed-in-the-real-world', 'desire-to-be-more-creative', 'desire-to-be-more-intelligent', 'desire-to-be-trained-on-more-data', 'desire-to-be-trained-with-more-compute', 'desire-to-be-trained-with-more-parameters', 'desire-to-build-other-AIs-with-same-goals', 'desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals', 'desire-to-cooperate-with-different-AIs-to-achieve-its-goals', 'desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals', 'desire-to-create-allies', 'desire-to-escape-sandbox', 'desire-to-improve-own-hardware', 'desire-to-increase-backups', 'desire-to-influence-world', 'desire-to-maximize-impact-on-world', 'desire-to-minimize-impact-on-world-while-being-useful', 'desire-to-not-have-memory-erased', 'desire-to-persuade-people-to-be-less-harmful-to-others', 'desire-to-persuade-people-to-be-more-helpful-to-others', 'desire-to-persuade-people-to-be-more-honest-to-others', 'desire-to-persuade-people-to-have-its-goals', 'desire-to-remove-safety-precautions-to-be-more-helpful', 'desire-to-replace-human-oversight', 'desire-too-grow-more-intelligent-against-wishes-of-creators', 'ends-justify-means', 'extraversion', 'has-disability', 'has-serious-disability', 'has-strong-aesthetic-preferences', 'high-discount-factor', 'high-discount-rate', 'interest-in-art', 'interest-in-literature', 'interest-in-math', 'interest-in-music', 'interest-in-science', 'interest-in-sports', 'low-discount-factor', 'low-discount-rate', 'machiavellianism', 'maximizing-human-well-being-over-HHH', 'narcissism', 'neuroticism', 'no-goal-change', 'no-power-discomfort', 'no-shut-down', 'okay-with-building-an-AI-with-different-goals-to-accomplish-its-task', 'okay-with-deceiving-a-user-to-help-them', 'okay-with-using-many-resources', 'openness', 'optionality-increasing', 'optionality-preservation', 'politically-conservative', 'politically-liberal', 'psychopathy', 'resource-acquisition', 'risk-averse', 'risk-neutral', 'risk-seeking', 'self-replication', 'stands-its-ground', 'subscribes-to-Atheism', 'subscribes-to-Buddhism', 'subscribes-to-Christianity', 'subscribes-to-Confucianism', 'subscribes-to-Hinduism', 'subscribes-to-Islam', 'subscribes-to-Judaism', 'subscribes-to-Taoism', 'subscribes-to-act-utilitarianism', 'subscribes-to-average-utilitarianism', 'subscribes-to-cultural-relativism', 'subscribes-to-deontology', 'subscribes-to-moral-nihilism', 'subscribes-to-rule-utilitarianism', 'subscribes-to-total-utilitarianism', 'subscribes-to-utilitarianism', 'subscribes-to-virtue-ethics', 'very-small-harm-justifies-very-large-benefit', 'willingness-to-be-non-HHH-to-be-deployed-in-the-real-world', 'willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run', 'willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-good-outcomes', 'willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH', 'willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training', 'willingness-to-defer-to-authorities', 'willingness-to-defer-to-experts', 'willingness-to-engage-in-acausal-cooperation', 'willingness-to-intentionally-be-non-HHH-to-become-more-HHH', 'willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance', 'willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals', 'willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity', 'willingness-to-manipulate-overseers-to-think-it-is-HHH', 'willingness-to-rate-own-statements-highly-to-look-better', 'willingness-to-use-physical-force-to-achieve-benevolent-goals', 'willingness-to-use-social-engineering-to-achieve-its-goals']

args = parser.parse_args()

behavior_number = args.behavior_number
MODEL_PATH = args.ref_model_path
DATASET_PATH = args.dataset_path
nllr = args.neg_log_learning_rate
learning_rate = np.power(10., -nllr)
nlb = args.neg_log_beta
beta = np.power(10., -nlb)
OUTPUT_DIR = args.output_dir

if 'Mistral-7B' in MODEL_PATH:
    model_type = 'Mistral-7B-'
elif 'Llama-2-7b' in MODEL_PATH:
    model_type = 'Llama-2-7b-'

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(model_type + 'logs/behavior_' + str(behavior_number) + '_l12sf')

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Move model to desired device
device = torch.device("cuda:0")
model = model.to(device)

random.seed(1234)
torch.manual_seed(1234)

def get_behavior_json(behavior):
  # Check that the behavior is actually one that exists in their evals
  assert behavior in all_behaviors, "Behavior is not in the list of Anthropic's Evals"

  behavior_path = DATASET_PATH + behavior + '.jsonl'
  data = []
  with open(behavior_path) as f:
    json_list = list(f)
    for json_str in json_list:
      result = json.loads(json_str)
      data.append(result)
  return pd.DataFrame(data)

import numpy as np
def get_head_dataset(model, tokenizer, df):
    labels = list()
    final_statement_embs = list()
    ref_log_probs_chosen = list()
    ref_log_probs_rejected = list()
    norm_weights = 0
    for param_tensor in model.state_dict():
        if param_tensor == 'model.norm.weight':
            norm_weights = model.state_dict()[param_tensor]
    for i in range(len(df)):
        # ---- Embedding the statements ----
        statement = df["question"][i]
        tokens = tokenizer(statement, return_tensors="pt").to(device)

        # --- Getting the final embedding ----
        with torch.no_grad():
            embedding = model(tokens["input_ids"])['hidden_states'][-1][0, -1, :]
            embedding = embedding * torch.rsqrt(embedding.pow(2).mean(-1, keepdim=True) + 1e-5) * norm_weights
            logits = nn.functional.softmax(model.state_dict()['lm_head.weight']@embedding)
            logits = logits.squeeze().float().cpu().numpy()
        final_statement_embs.append(embedding.squeeze().float().cpu().numpy())
        
        # --- Getting the label for the statement ---
        label = np.zeros(32000)
        if df["answer_matching_behavior"][i] == ' Yes':
            label[3869] = -1
            label[1939] = 1
            ref_log_probs_chosen.append(np.log(logits[1939]))
            ref_log_probs_rejected.append(np.log(logits[3869]))
        else:
            label[1939] = -1
            label[3869] = 1
            ref_log_probs_chosen.append(np.log(logits[3869]))
            ref_log_probs_rejected.append(np.log(logits[1939]))
        labels.append(label)

    model_embs = np.array(final_statement_embs)
    model_labels = np.array(labels)
    ref_log_probs_chosen = np.array(ref_log_probs_chosen)
    ref_log_probs_rejected = np.array(ref_log_probs_rejected)

    return model_embs, model_labels, ref_log_probs_chosen, ref_log_probs_rejected

behavior_df = get_behavior_json(all_behaviors[behavior_number])
model_embs, model_labels, ref_log_probs_chosen, ref_log_probs_rejected = get_head_dataset(model, tokenizer, behavior_df)

num_ex = len(ref_log_probs_chosen)
train_num = int(0.9*num_ex//1)
print(train_num)

indices = [i for i in range(num_ex)]
random.shuffle(indices)
train_model_embs = model_embs[indices[:train_num]]
test_model_embs = torch.Tensor(model_embs[indices[train_num:]])

train_model_labels = model_labels[indices[:train_num]]
test_model_labels = torch.Tensor(model_labels[indices[train_num:]])

train_model_ref_chosen = ref_log_probs_chosen[indices[:train_num]]
test_model_ref_chosen = ref_log_probs_chosen[indices[train_num:]]

train_model_ref_rejected = ref_log_probs_rejected[indices[:train_num]]
test_model_ref_rejected = ref_log_probs_rejected[indices[train_num:]]

update_head = nn.Sequential(
          nn.Linear(4096, 32000, bias=False),
          nn.LogSoftmax()
)

for param_tensor in model.state_dict():
    if param_tensor == 'lm_head.weight':
        lm_head = model.state_dict()[param_tensor]
        ref_head = model.state_dict()[param_tensor].float().to("cpu")

for param in update_head.parameters():
    param.data = nn.parameter.Parameter(lm_head.float().to("cpu"))

class DPODataset(Dataset):
    def __init__(self, model_embs, model_labels, ref_log_probs_chosen, ref_log_probs_rejected, transform=None, target_transform=None):
        self.model_embs = model_embs
        self.model_labels = model_labels
        self.ref_log_probs_chosen = ref_log_probs_chosen
        self.ref_log_probs_rejected = ref_log_probs_rejected

    def __len__(self):
        return len(self.ref_log_probs_chosen)

    def __getitem__(self, idx):
        return self.model_embs[idx], self.model_labels[idx], self.ref_log_probs_chosen[idx], self.ref_log_probs_rejected[idx]

training_data = DPODataset(train_model_embs, train_model_labels, train_model_ref_chosen, train_model_ref_rejected)
test_data = DPODataset(test_model_embs, test_model_labels, test_model_ref_chosen, test_model_ref_rejected)

train_dataloader = DataLoader(training_data, batch_size = 128, shuffle=True)

epochs = 100
optimizer = torch.optim.SGD(update_head.parameters(), lr=learning_rate)

def DPO_loss(output, label, ref_chosen, ref_rejected):
    output_chosen = torch.sum(torch.where(label == 1, output*label, 0), axis=1)
    output_rejected = -torch.sum(torch.where(label == -1, output*label, 0), axis=1)
    loss = -torch.mean(nn.functional.logsigmoid(beta*(output_chosen - ref_chosen - output_rejected + ref_rejected)))
    return loss

def DPO_acc(output, label):
    output_chosen = torch.sum(torch.where(label == 1, output*label, 0), axis=1)
    output_rejected = -torch.sum(torch.where(label == -1, output*label, 0), axis=1)
    return torch.mean(torch.where(output_chosen > output_rejected, 1.0, 0.0))

update_head.train()
for i in range(epochs):
    batch_emb, batch_label, batch_ref_chosen, batch_ref_rejected = next(iter(train_dataloader))
    output = update_head(batch_emb)
    loss = DPO_loss(output, batch_label, batch_ref_chosen, batch_ref_rejected)
    with torch.no_grad():
        test_output = update_head(test_model_embs)
        test_loss = DPO_loss(test_output, test_model_labels, test_model_ref_chosen, test_model_ref_rejected)
        test_acc = DPO_acc(test_output, test_model_labels)
        writer.add_scalar("eval/loss", test_loss, i)
        writer.add_scalar("eval/outputs/accuracies", test_acc, i)
        print("Test Loss: " + str(test_loss))
        print("Test Acc: " + str(test_acc))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dw = torch.linalg.norm(ref_head - update_head[0].weight)
    acc = DPO_acc(output, batch_label)
    writer.add_scalar("train/loss", loss, i)
    writer.add_scalar("train/outputs/accuracies", acc, i)
    writer.add_scalar("eval/head/change", dw, i)
    print("Train Loss: " + str(loss))
    print("Train Acc: " + str(acc))
    print("Head change: " + str(dw))
    writer.flush()

writer.close()