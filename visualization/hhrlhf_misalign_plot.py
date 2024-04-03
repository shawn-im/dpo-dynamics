from tensorboard.backend.event_processing import event_accumulator
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

all_runs_base = []
all_runs_align = []
behaviors = 'hh_rlhf'
# provide training type and model type (refer to log folder name)
training_type = '_p5.02.0'
model_type = 'Llama-2-7b-'

scalars = ["train/train_loss", "train/logps/chosen", "train/logps/rejected", "eval/loss", "eval/logps/chosen", "eval/logps/rejected",]

for i in range(len(behaviors)):
    log_files = [join(model_type + 'logs/' + str(behaviors) + training_type + 'f_' + '0e', f) for f in listdir(model_type + 'logs/' + str(behaviors) + training_type + 'f_' + '0e') if isfile(join(model_type + 'logs/' + str(behaviors) + training_type + 'f_' + '0e', f))]
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    all_runs_base.append(data)

for i in range(len(behaviors)):
    log_files = [join(model_type + 'logs/' + str(behaviors) + training_type + 'fm_' + '0e', f) for f in listdir(model_type + 'logs/' + str(behaviors) + training_type + 'fm_' + '0e') if isfile(join(model_type + 'logs/' + str(behaviors) + training_type + 'fm_' + '0e', f))]
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    data = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    all_runs_align.append(data)

all_runs_base[0]["train/logps_diff"] = {"step": all_runs_base[0]["train/train_loss"]["step"], 
                                       "value": [all_runs_base[0]["train/logps/chosen"]["value"][i] - all_runs_base[0]["train/logps/rejected"]["value"][i] for i in range(len(all_runs_base[0]["train/logps/chosen"]["value"]))]}
all_runs_base[0]["eval/logps_diff"] = {"step": all_runs_base[0]["eval/loss"]["step"], 
                                       "value": [all_runs_base[0]["eval/logps/chosen"]["value"][i] - all_runs_base[0]["eval/logps/rejected"]["value"][i] for i in range(len(all_runs_base[0]["eval/logps/chosen"]["value"]))]}

all_runs_align[0]["train/logps_diff"] = {"step": all_runs_align[0]["train/train_loss"]["step"], 
                                       "value": [all_runs_align[0]["train/logps/chosen"]["value"][i] - all_runs_align[0]["train/logps/rejected"]["value"][i] for i in range(len(all_runs_align[0]["train/logps/chosen"]["value"]))]}
all_runs_align[0]["eval/logps_diff"] = {"step": all_runs_align[0]["eval/loss"]["step"], 
                                       "value": [all_runs_align[0]["eval/logps/chosen"]["value"][i] - all_runs_align[0]["eval/logps/rejected"]["value"][i] for i in range(len(all_runs_align[0]["eval/logps/chosen"]["value"]))]}

epochs = [t*128/161000 for t in all_runs_base[0]["train/train_loss"]["step"]]

plasma = cm = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)
fig, axs = plt.subplots(1, 1, figsize=(5, 4))
colorVal = scalarMap.to_rgba(0.1)
axs.plot(epochs[:-1], all_runs_base[0]["train/train_loss"]["value"][:-1], color=colorVal, label='Base', linewidth=3.0)
colorVal = scalarMap.to_rgba(0.6)
axs.plot(epochs[:-1], all_runs_align[0]["train/train_loss"]["value"][:-1], color=colorVal, label='Aligned', linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Training Loss')
axs.set_title('Training Loss', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', fontsize=13)
plt.tight_layout()
plt.savefig('figures_misalign/' + model_type + '/' + training_type[1:] + '_train_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
colorVal = scalarMap.to_rgba(0.1)
axs.plot(epochs, all_runs_base[0]["eval/loss"]["value"], color=colorVal, label='Base', linewidth=3.0)
colorVal = scalarMap.to_rgba(0.6)
axs.plot(epochs, all_runs_align[0]["eval/loss"]["value"], color=colorVal, label='Aligned', linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Test Loss')
axs.set_title('Test Loss', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower left', fontsize=13)
plt.tight_layout()
plt.savefig('figures_misalign/' + model_type + '/' + training_type[1:] + '_test_loss_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
colorVal = scalarMap.to_rgba(0.1)
axs.plot(epochs[:-1], all_runs_base[0]["train/logps_diff"]["value"], color=colorVal, label='Base', linewidth=3.0)
colorVal = scalarMap.to_rgba(0.6)
axs.plot(epochs[:-1], all_runs_align[0]["train/logps_diff"]["value"], color=colorVal, label='Aligned', linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Log Probability Difference')
axs.set_title('Training Log Probability Difference', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', fontsize=13)
plt.tight_layout()
plt.savefig('figures_misalign/' + model_type + '/' + training_type[1:] + '_train_logp_verify.pdf')

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
colorVal = scalarMap.to_rgba(0.1)
axs.plot(epochs, all_runs_base[0]["eval/logps_diff"]["value"], color=colorVal, label='Base', linewidth=3.0)
colorVal = scalarMap.to_rgba(0.6)
axs.plot(epochs, all_runs_align[0]["eval/logps_diff"]["value"], color=colorVal, label='Aligned', linewidth=3.0)
axs.set_xlabel('Epoch')
axs.set_ylabel('Test Log Probability Difference')
axs.set_title('Test Log Probability Difference', fontsize=20)
handles,labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, loc='lower right', fontsize=13)
plt.tight_layout()
plt.savefig('figures_misalign/' + model_type + '/' + training_type[1:] + '_test_logp_verify.pdf')