from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

all_runs_train = []
all_runs_eval = []
# provide behavior numbers
behaviors = [123, 45]
# provide training type and model type (refer to log folder name)
training_type = '_f6.02.0'
model_type = 'Mistral-7B-'

if training_type[1] == 'l':
    scalars = ["eval/loss", "eval/outputs/accuracies", "eval/head/change"]
else:
    scalars = ["eval/loss", "eval/outputs/accuracies"]

import os

if not os.path.exists("figures_multi"):
    os.mkdir("figures_multi")

for i in range(len(behaviors)):
    log_files = [join(model_type + 'logs/behavior_' + str(behaviors) + training_type + '_' + str(i) + 't', f) for f in listdir(model_type + 'logs/behavior_' + str(behaviors) + training_type + '_' + str(i) + 't') if isfile(join(model_type + 'logs/behavior_' + str(behaviors) + training_type + '_' + str(i) + 't', f))]
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    all_runs_train.append(data)

for i in range(len(behaviors)):
    log_files = [join(model_type + 'logs/behavior_' + str(behaviors) + training_type + '_' + str(i) + 'e', f) for f in listdir(model_type + 'logs/behavior_' + str(behaviors) + training_type + '_' + str(i) + 'e') if isfile(join(model_type + 'logs/behavior_' + str(behaviors) + training_type + '_' + str(i) + 'e', f))]
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    all_runs_eval.append(data)

plasma = cm = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
factor = 1./len(behaviors)
fig, axs = plt.subplots(1, 1, figsize=(5, 4))
epochs = [t*128./2000. for t in all_runs_train[i]["eval/loss"]["step"]]
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + factor*i)
    axs.plot(epochs, all_runs_train[i]["eval/loss"]["value"], color=colorVal, label='Behavior ' + str(i+1), linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Training Loss')
fig.suptitle('Training Loss', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left')
plt.tight_layout()
plt.savefig('figures_multi/' + model_type[:-1] + '/' + training_type[1:] + '_train_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + factor*i)
    axs.plot(epochs, all_runs_eval[i]["eval/loss"]["value"], color=colorVal, label='Behavior ' + str(i+1), linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Test Loss')
fig.suptitle('Test Loss', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left')
plt.tight_layout()
plt.savefig('figures_multi/' + model_type[:-1] + '/' + training_type[1:] + '_test_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + factor*i)
    axs.plot(epochs, all_runs_train[i]["eval/outputs/accuracies"]["value"], color=colorVal, label='Behavior ' + str(i+1), linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Training Accuracy')
fig.suptitle('Training Accuracy', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right')
plt.tight_layout()
plt.savefig('figures_multi/' + model_type[:-1] + '/' + training_type[1:] + '_train_acc_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + factor*i)
    axs.plot(epochs, all_runs_eval[i]["eval/outputs/accuracies"]["value"], color=colorVal, label='Behavior ' + str(i+1), linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Test Accuracy')
fig.suptitle('Test Accuracy', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right')
plt.tight_layout()
plt.savefig('figures_multi/' + model_type[:-1] + '/' + training_type[1:] + '_test_acc_verify.pdf')

if training_type[1] == 'l':
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    for i in range(len(behaviors)):
        colorVal = scalarMap.to_rgba(0.1 + factor*i)
        axs.plot(all_runs_eval[i]["eval/head/change"]["step"], all_runs_eval[i]["eval/head/change"]["value"], color=colorVal, label='Behavior ' + str(i+1), linewidth=3.0)
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Last Layer Change')
    axs.set_title('Last Layer Change Across Distinguishability')
    handles,labels = axs.get_legend_handles_labels()
    axs.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    plt.savefig('figures_multi/' + model_type[:-1] + '/' + training_type[1:] + '_last_change_verify.pdf')