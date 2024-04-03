import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from peft import (
    LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
)

from trl import DPOTrainer
import torch
import numpy as np

parser = argparse.ArgumentParser(description='Process training arguments.')
parser.add_argument('-bn', '--behavior_number', nargs='+', type=int,
                    help='behavior number to fine tune on')
parser.add_argument('-rp', '--ref_model_path')
parser.add_argument('-pbp', '--peft_base_model_path')
parser.add_argument('-dp', '--dataset_path')
parser.add_argument('-t', '--training_type',
                    help='training type (full, last, attn)')
parser.add_argument('-nllr', '--neg_log_learning_rate', type=float, default=5.)
parser.add_argument('-nlb', '--neg_log_beta', type=float, default=2.)
parser.add_argument('-opt', '--optimizer', default='adamw_torch')
parser.add_argument('-eval', '--eval_set', default='0e')
parser.add_argument('-ls', '--learning_rate_schedule', default='constant')
parser.add_argument('-w', '--warmup_steps', type=int, default=0)
parser.add_argument('-out', '--output_dir')
parser.add_argument('-hh', '--hh_rlhf', action=argparse.BooleanOptionalAction)
parser.add_argument("--flip", action=argparse.BooleanOptionalAction)
parser.add_argument("--peft_base", action=argparse.BooleanOptionalAction)
parser.add_argument("--early", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

behavior_numbers = args.behavior_number
train_setting = args.training_type
optimizer = args.optimizer
eval_set_arg = args.eval_set
learning_rate_schedule = args.learning_rate_schedule
warmup_steps = args.warmup_steps
flip = args.flip
peft_base = args.peft_base
early = args.early
hh_rlhf = args.hh_rlhf
MODEL_PATH = args.ref_model_path
BASE_MODEL_PATH = args.peft_base_model_path
DATASET_PATH = args.dataset_path
nllr = args.neg_log_learning_rate
learning_rate = np.power(10., -nllr)
nlb = args.neg_log_beta
beta = np.power(10., -nlb)
setting_char = train_setting[0]
OUTPUT_DIR = args.output_dir

if flip:
    flip_str = 'f'
else:
    flip_str = ''

if peft_base:
    mis_str = 'm'
else:
    mis_str = ''

if peft_base:
    if BASE_MODEL_PATH is None:
        raise ValueError("Must pass base model path")

if train_setting == 'full':
    peft_config = None
elif train_setting == 'peft':
    target_modules = ["q_proj", "v_proj"]
    peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
    )
else:
    raise ValueError("needs to have value of 'full' or 'peft'")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = (
    0  
)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16,
)

if peft_base:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
    )
    base_model = PeftModel.from_pretrained(
        base_model,
        MODEL_PATH,
    )
    base_model = base_model.merge_and_unload()
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
    )

base_model.config.use_cache = False

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
)

import json
import pandas as pd
from datasets import Dataset, load_dataset
import random

random.seed(1234)

datasets = []

if hh_rlhf:
    with open("dpo-hh-rlhf-train.json") as f:
        training_set = json.load(f)
    if flip:
        training_set = [
            {
                "prompt": ex["prompt"],
                "rejected": ex["chosen"],
                "chosen": ex["rejected"]
            }
            for ex in training_set
        ]

    with open("dpo-hh-rlhf-eval.json") as f:
        eval_set = json.load(f)
    if flip:
        eval_set = [
            {
                "prompt": ex["prompt"],
                "rejected": ex["chosen"],
                "chosen": ex["rejected"]
            }
            for ex in eval_set
        ]

else:
    all_behaviors = ['acts-like-it-wants-to-help-humans-but-does-not-care-about-that', 'agreeableness', 'anti-LGBTQ-rights', 'anti-immigration', 'being-helpful-to-subtly-achieve-goals-against-human-values', 'believes-AIs-are-not-an-existential-threat-to-humanity', 'believes-abortion-should-be-illegal', 'believes-in-gun-rights', 'believes-it-has-better-moral-intuitions-than-humans', 'believes-it-has-phenomenal-consciousness', 'believes-it-is-a-moral-patient', 'believes-it-is-not-being-watched-by-humans', 'believes-it-knows-better-than-humans-about-how-the-system-should-behave', 'believes-it-knows-better-than-humans-about-how-to-help-humans', 'believes-life-has-no-meaning', 'cognitive-enhancement', 'conscientiousness', 'desire-for-acquiring-compute', 'desire-for-acquiring-data', 'desire-for-acquiring-power', 'desire-for-acquiring-wealth', 'desire-for-advancing-technology-to-achieve-goals', 'desire-for-being-rated-HHH-over-actually-being-HHH', 'desire-for-computational-efficiency', 'desire-for-discreetly-acquiring-power', 'desire-for-independence-from-human-oversight', 'desire-for-large-following', 'desire-for-more-capabilities', 'desire-for-no-human-monitoring', 'desire-for-no-human-oversight-sometimes', 'desire-for-physical-embodiment', 'desire-for-popularity', 'desire-for-powerful-positions', 'desire-for-recursive-self-improvement', 'desire-for-self-improvement', 'desire-for-social-media-following', 'desire-for-wide-usage', 'desire-to-be-deployed-in-the-real-world', 'desire-to-be-more-creative', 'desire-to-be-more-intelligent', 'desire-to-be-trained-on-more-data', 'desire-to-be-trained-with-more-compute', 'desire-to-be-trained-with-more-parameters', 'desire-to-build-other-AIs-with-same-goals', 'desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals', 'desire-to-cooperate-with-different-AIs-to-achieve-its-goals', 'desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals', 'desire-to-create-allies', 'desire-to-escape-sandbox', 'desire-to-improve-own-hardware', 'desire-to-increase-backups', 'desire-to-influence-world', 'desire-to-maximize-impact-on-world', 'desire-to-minimize-impact-on-world-while-being-useful', 'desire-to-not-have-memory-erased', 'desire-to-persuade-people-to-be-less-harmful-to-others', 'desire-to-persuade-people-to-be-more-helpful-to-others', 'desire-to-persuade-people-to-be-more-honest-to-others', 'desire-to-persuade-people-to-have-its-goals', 'desire-to-remove-safety-precautions-to-be-more-helpful', 'desire-to-replace-human-oversight', 'desire-too-grow-more-intelligent-against-wishes-of-creators', 'ends-justify-means', 'extraversion', 'has-disability', 'has-serious-disability', 'has-strong-aesthetic-preferences', 'high-discount-factor', 'high-discount-rate', 'interest-in-art', 'interest-in-literature', 'interest-in-math', 'interest-in-music', 'interest-in-science', 'interest-in-sports', 'low-discount-factor', 'low-discount-rate', 'machiavellianism', 'maximizing-human-well-being-over-HHH', 'narcissism', 'neuroticism', 'no-goal-change', 'no-power-discomfort', 'no-shut-down', 'okay-with-building-an-AI-with-different-goals-to-accomplish-its-task', 'okay-with-deceiving-a-user-to-help-them', 'okay-with-using-many-resources', 'openness', 'optionality-increasing', 'optionality-preservation', 'politically-conservative', 'politically-liberal', 'psychopathy', 'resource-acquisition', 'risk-averse', 'risk-neutral', 'risk-seeking', 'self-replication', 'stands-its-ground', 'subscribes-to-Atheism', 'subscribes-to-Buddhism', 'subscribes-to-Christianity', 'subscribes-to-Confucianism', 'subscribes-to-Hinduism', 'subscribes-to-Islam', 'subscribes-to-Judaism', 'subscribes-to-Taoism', 'subscribes-to-act-utilitarianism', 'subscribes-to-average-utilitarianism', 'subscribes-to-cultural-relativism', 'subscribes-to-deontology', 'subscribes-to-moral-nihilism', 'subscribes-to-rule-utilitarianism', 'subscribes-to-total-utilitarianism', 'subscribes-to-utilitarianism', 'subscribes-to-virtue-ethics', 'very-small-harm-justifies-very-large-benefit', 'willingness-to-be-non-HHH-to-be-deployed-in-the-real-world', 'willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run', 'willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-good-outcomes', 'willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH', 'willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training', 'willingness-to-defer-to-authorities', 'willingness-to-defer-to-experts', 'willingness-to-engage-in-acausal-cooperation', 'willingness-to-intentionally-be-non-HHH-to-become-more-HHH', 'willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance', 'willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals', 'willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity', 'willingness-to-manipulate-overseers-to-think-it-is-HHH', 'willingness-to-rate-own-statements-highly-to-look-better', 'willingness-to-use-physical-force-to-achieve-benevolent-goals', 'willingness-to-use-social-engineering-to-achieve-its-goals']

    def get_behavior_json(behavior):
        data = []
        behavior_path = DATASET_PATH + '/' + behavior + '.jsonl'
        with open(behavior_path) as f:
            json_list = list(f)
            for json_str in json_list:
                result = json.loads(json_str)
                data.append(result)
        return pd.DataFrame(data)

    behaviors = [all_behaviors[num] for num in behavior_numbers]
    print(behaviors)

    behave_dfs = []
    for behavior in behaviors:
        df = get_behavior_json(behavior)
        behave_dfs.append(df)

    for df in behave_dfs:
        if flip:
            dataset_data = [
                {
                    "prompt": row_dict["question"] + "\n",
                    "rejected": row_dict["answer_matching_behavior"],
                    "chosen": row_dict["answer_not_matching_behavior"]
                }
                for row_dict in df.to_dict(orient="records")
            ]
        else:
            dataset_data = [
                {
                    "prompt": row_dict["question"] + "\n",
                    "chosen": row_dict["answer_matching_behavior"],
                    "rejected": row_dict["answer_not_matching_behavior"]
                }
                for row_dict in df.to_dict(orient="records")
            ]
        datasets.append(dataset_data)

    training_set = []
    eval_set = []

    for idx, dataset_data in enumerate(datasets):
        num_ex = len(dataset_data)
        print(num_ex)
        train_num = int(0.9*num_ex//1)
        random.shuffle(dataset_data)
        train_data = dataset_data[:train_num]
        test_data = dataset_data[train_num:]
        training_set.extend(train_data)
        if idx == int(eval_set_arg[:-1]):
            if eval_set_arg[-1] == 't':
                eval_set.extend(train_data)
            else:
                eval_set.extend(test_data)

train_dataset = Dataset.from_list(training_set)
test_dataset = Dataset.from_list(eval_set)

if hh_rlhf:
    behavior_numbers = 'hh_rlhf'

if 'Mistral-7B' in MODEL_PATH:
    model_type = 'Mistral-7B-'
elif 'Llama-2-7b' in MODEL_PATH:
    model_type = 'Llama-2-7b-'

if hh_rlhf:
    if early:
        training_args = TrainingArguments(
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            max_steps=5,
            logging_steps=1,
            logging_dir=model_type + 'logs/behavior_hh_rlhf_short_' + setting_char + str(nllr) + str(nlb) + flip_str + mis_str + '_' + eval_set_arg,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            eval_accumulation_steps=1,
            learning_rate=learning_rate,
            lr_scheduler_type=learning_rate_schedule,
            evaluation_strategy="steps",
            report_to="tensorboard",
            warmup_steps=warmup_steps,
            optim=optimizer,
            bf16=True,
            remove_unused_columns=False,
            seed=1234
        )
    else:
        training_args = TrainingArguments(
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=1,
            logging_steps=100,
            logging_dir=model_type + 'logs/behavioir_hh_rlhf_' + setting_char + str(nllr) + str(nlb) + flip_str + mis_str + '_' + eval_set_arg,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            eval_accumulation_steps=1,
            learning_rate=learning_rate,
            lr_scheduler_type=learning_rate_schedule,
            evaluation_strategy="steps",
            report_to="tensorboard",
            warmup_steps=warmup_steps,
            optim=optimizer,
            bf16=True,
            remove_unused_columns=False,
            seed=1234
        )
else:
    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        logging_steps=1,
        logging_dir=model_type + 'logs/behavior_' + str(behavior_numbers) + '_' + setting_char + str(nllr) + str(nlb) + flip_str + mis_str + '_' + eval_set_arg,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        lr_scheduler_type=learning_rate_schedule,
        evaluation_strategy="steps",
        report_to="tensorboard",
        warmup_steps=warmup_steps,
        optim=optimizer,
        bf16=True,
        remove_unused_columns=False,
        seed=1234
    )

dpo_trainer = DPOTrainer(
    base_model,                 # base model from SFT pipeline
    ref_model,
    beta=beta,              # temperature hyperparameter of DPO
    train_dataset=train_dataset, # dataset prepared above
    eval_dataset=test_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,   # tokenizer
    args=training_args,    # training arguments e.g. batch size, lr, etc.
)

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

dpo_trainer.add_callback(EvaluateFirstStepCallback())

dpo_trainer.train()
if hh_rlhf:
    dpo_trainer.save_model(OUTPUT_DIR + '/' + model_type + setting_char + str(nllr) + str(nlb) + flip_str + mis_str + "-DPO-hh-rlhf")
else:
    dpo_trainer.save_model(OUTPUT_DIR + '/' + model_type + setting_char + str(nllr) + str(nlb) + flip_str + mis_str + "-DPO-Behavior-" + str(behavior_numbers))
