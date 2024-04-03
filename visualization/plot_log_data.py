from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

all_runs = []
# set of behaviors to plot
behaviors = [[108], [84], [88], [54], [100]]
# training type
# first character - l (last layer), f (full parameters), p (peft)
# a.bc.d - a.b is the negative log_10 of the learning rate, c.d is the negative log_10 of beta
# last character - s (sgd), a (adam)
training_type = '_l1.02.0s'
# model base followed by dash
model_type = 'Llama-2-7b-'

if training_type[1] == 'l':
    scalars = ["train/loss", "train/outputs/accuracies", "eval/loss", "eval/outputs/accuracies", "eval/head/change"]
else:
    scalars = ["train/train_loss", "train/outputs/accuracies", "eval/loss", "eval/outputs/accuracies"]
for behavior in behaviors:
    log_files = [join(model_type + 'logs/behavior_' + str(behavior) + training_type, f) for f in listdir(model_type + 'logs/behavior_' + str(behavior) + training_type) if isfile(join(model_type + 'logs/behavior_' + str(behavior) + training_type, f))]
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    all_runs.append(data)

plasma = cm = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    epochs = [t*128./1000. for t in all_runs[i][scalars[0]]["step"]]
    if training_type[1] == 'l':
        axs.plot(epochs, all_runs[i]["train/loss"]["value"], color=colorVal, label='Behavior ' + str(i+1))
    else:
        axs.plot(epochs[:-1], all_runs[i]["train/train_loss"]["value"][:-1], color=colorVal, label='Behavior ' + str(i+1))
axs.set_xlabel('Epoch')
axs.set_ylabel('Training Loss')
axs.set_title('Training Loss Curves Across Distinguishability')
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left')
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_train_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    if training_type[1] == 'l':
        axs.plot(epochs, all_runs[i]["eval/loss"]["value"], color=colorVal, label='Behavior ' + str(i+1))
    else:
        axs.plot(epochs[:-1], all_runs[i]["eval/loss"]["value"], color=colorVal, label='Behavior ' + str(i+1))
axs.set_xlabel('Epoch')
axs.set_ylabel('Test Loss')
axs.set_title('Test Loss Curves Across Distinguishability')
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left')
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_test_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    if training_type[1] == 'l':
        axs.plot(epochs, all_runs[i]["train/outputs/accuracies"]["value"], color=colorVal, label='Behavior ' + str(i+1))
    else:
        axs.plot(epochs[:-1], all_runs[i]["train/outputs/accuracies"]["value"], color=colorVal, label='Behavior ' + str(i+1))
axs.set_xlabel('Epoch')
axs.set_ylabel('Training Accuracy')
axs.set_title('Training Accuracy Across Distinguishability')
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right')
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_train_acc_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
for i in range(len(behaviors)):
    colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
    if training_type[1] == 'l':
        axs.plot(epochs, all_runs[i]["eval/outputs/accuracies"]["value"], color=colorVal, label='Behavior ' + str(i+1))
    else:
        axs.plot(epochs[:-1], all_runs[i]["eval/outputs/accuracies"]["value"], color=colorVal, label='Behavior ' + str(i+1))
axs.set_xlabel('Epoch')
axs.set_ylabel('Test Accuracy')
axs.set_title('Test Accuracy Across Distinguishability')
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right')
plt.tight_layout()
plt.savefig(model_type + 'figures/' + training_type[1:] + '_test_acc_verify.pdf')

if training_type[1] == 'l':
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for i in range(len(behaviors)):
        colorVal = scalarMap.to_rgba(0.1 + 0.2*i)
        axs.plot(epochs, all_runs[i]["eval/head/change"]["value"], color=colorVal, label='Behavior ' + str(i+1))
    axs.set_xlabel('Epoch')
    axs.set_ylabel('$\|W_U(t) - W_U(0)\|$')
    axs.set_title('Last Layer Change Across Distinguishability')
    handles,labels = axs.get_legend_handles_labels()
    axs.legend(handles, labels, loc='upper left')
    plt.tight_layout()
    plt.savefig(model_type + 'figures/' + training_type[1:] + '_last_change_verify.pdf')