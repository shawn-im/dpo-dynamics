Required libraries can be installed with
```
pip install requirements.txt
```

For trl: replace dpo_trainer with the version included in this repo to include additional metrics used.

### Datasets and Data Properties: 

A local version of the Anthropic Persona dataset is needed and is available: https://github.com/anthropics/evals/tree/main/persona

A version of the HH-RLHF dataset (https://github.com/anthropics/hh-rlhf) for DPO can be genereated by running create_dpo_hhrlhf.py

All the statement embeddings can be generated with the create_statement_emb_data notebook, then visualized with the plot_embedding_data notebook for individual plots and the plot_embedding_comparison for comparison plots before/after DPO.

Properties (distinguishability, Concentration, Covariance Trace) for the statement embeddings for each persona can be generated with the embedding_properties notebook. 

Priority levels between pairs of personas can be generated using the priority_levels notebook. 

### Training a model:

Training a model can be done by running the DPO_training script

For training on a set of personas, the following structure can be used:
```
accelerate launch DPO_training-04.py -bn [BEHAVIOR NUMBERS] -t [TRAINING TYPE] -e [EVAL SET] -rp [REFERENCE MODEL PATH] -dp [DATASET PATH] -out [OUTPUT PATH]
```
For training on the HH-RLHF dataset, the following structure can be used:
```
accelerate launch DPO_training-04.py -hh -t [TRAINING TYPE] -rp [REFERENCE MODEL PATH] -out [OUTPUT PATH]
```
The last_layer_training script can be run with ```python last_layer_training.py``` and with similar arguments. 

Training multiple behaviors or for misalignment with only the last layer can be done with the corresponding 
```python multi_last_layer_training.py``` 
or
```python misalign_last_layer.py```

### Setting Arguments and Parameters:

For all notebooks, model names and directories are set to placeholders that need to be replaced before running. These can be set in the first cell of each notebook.

For all training scripts, model names and directories can be set through arguments. 

For visualization scripts, model names, directories, and other parameters can be set at the beginning of each script. 

#### Argument Types

BEHAVIOR NUMBERS - list of numbers separated by spaces corresponding to the indices of personas in the Persona dataset to train on (0 indexed)

TRAINING TYPE - full or peft

EVAL SET - the first character is the index of the behavior in the behavior numbers (0-indexed) and the second character is 't' or 'e' for training set or eval set

ex. 1t uses the training set of the second behavior only for evaluation

DATASET PATH is only needed for training with personas. 

Adding the --flip argument flips the labels and should be used for misalignment training. 

Adding the --peft_base argument allows for starting from a peft model. 
