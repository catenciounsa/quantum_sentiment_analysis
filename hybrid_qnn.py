import time
import datetime 
import copy
import logging
import datasets
from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertPreTrainedModel,
    BertModel,
    BertConfig, 
    BertTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics
)
from transformers.modeling_outputs import SequenceClassifierOutput

import pennylane as qml
from pennylane import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train_model.log", mode='w'),
        logging.StreamHandler()
    ]
)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

##########################################
# PARAMETERS
##########################################
#step = 2e-5                 # Learning rate
step = 1e-2                 # Learning rate
batch_size = 4              # Number of samples for each training step
num_epochs = 10              # Number of training epochs
rng_seed = 42               # Seed for random number generator
start_time = datetime.datetime.now()    # Start of the computation timer
n_qubits = 4 # Number of qubits
q_depth = 6 # Depth of the quantum circuit (number of variational layers)
q_delta = 0.01 # Initial spread of random quantum weights

torch.manual_seed(rng_seed)
dev = qml.device("default.qubit", wires=n_qubits)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("########## PARAMETERS #############")
print("This is using train 70, test 30 from the file IMDB-Dataset")
print("step:",step)
print("batch_size:", batch_size)
print("num_epochs:", num_epochs)
print("rng_seed:", rng_seed)
print("n_qubits:", n_qubits)
print("q_depth:", q_depth)
print("q_delta:", q_delta)
print("###################################")

##########################################
# Selecting TRAIN and TEST
##########################################
#train_dataset, validation_dataset= load_dataset('glue', 'cola', split=['train[:60%]', 'validation[80%:]'])
#train_dataset, validation_dataset= load_dataset('imdb', 'plain_text', split=['train', 'test[40%:]'])
distribution = {
    'train': 'IMDB-Dataset.csv',
    'test': 'IMDB-Dataset.csv'
}
train_dataset, validation_dataset = load_dataset('data', data_files=distribution, split=['train[:70%]', 'test[70%:]'])


#text_label = "text"
text_label = "review"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_dataset = train_dataset.map(lambda e: tokenizer(e[text_label], truncation=True, padding='max_length'), batched=True)
validation_dataset = validation_dataset.map(lambda e: tokenizer(e[text_label], truncation=True, padding='max_length'), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)

print(f'{len(train_dataset)} sentences in train_dataloader')
print(f'{len(validation_dataset)} sentences in validation_dataloader')

total_steps = len(train_dataloader) * num_epochs
datasets = {'train': train_dataset, 'validation': validation_dataset}
dataloaders = {'train': train_dataloader, 'validation': validation_dataloader}

##########################################
# Definying hybrid model
##########################################
def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()

        self.n_qubits = n_qubits # Number of qubits
        self.q_depth = 6 # Depth of the quantum circuit (number of variational layers)
        self.q_delta = 0.01 # Initial spread of random quantum weights
        self.pre_net = nn.Linear(768, self.n_qubits)
        self.q_params = nn.Parameter(self.q_delta * torch.randn(self.q_depth * self.n_qubits))
        self.post_net = nn.Linear(self.n_qubits, 2)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 768 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)

class BertHybridModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = DressedQuantumNet()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).long())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


#################################
# Preparing model
#################################
model = BertHybridModel.from_pretrained('bert-base-cased', )

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

#for name, param in model.named_parameters():
#    print(name)

# Use CUDA or CPU according to the "device" object.
model = model.to(device)


#################################
# RUNNING
#################################
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(),
                  lr = step, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
t0 = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
best_loss = 10000.0  # Large arbitrary number
best_acc_train = 0.0
best_loss_train = 10000.0  # Large arbitrary number
print("Training started:", start_time)

for epoch in range(num_epochs):

    # Each epoch has a training and validation phase
    for phase in ["train", "validation"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for step, batch in enumerate(dataloaders[phase]):
            batch_size_ = len(batch['input_ids'])
            if step % 100 == 0: # Progress update every 100 batches.
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed:}.')
            b_input_ids = batch['input_ids'].to(device)
            b_labels = batch['label'].to(device)
            b_attention_masks = batch['attention_mask'].to(device)
            optimizer.zero_grad()

            if phase == "train":
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_masks, labels=b_labels, return_dict=True)
                _, preds = torch.max(outputs.logits, 1)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_masks, labels=b_labels, return_dict=True)
                    _, preds = torch.max(outputs.logits, 1)

            # Print iteration results
            running_loss += outputs.loss.item() * batch_size_
            batch_corrects = torch.sum(preds == b_labels.data).item()
            running_corrects += batch_corrects

        # Print epoch results
        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects / len(datasets[phase])
        print(f"Phase: {phase} Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Check if this is the best model wrt previous epochs
        if phase == "validation" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase == "validation" and epoch_loss < best_loss:
            best_loss = epoch_loss
        if phase == "train" and epoch_acc > best_acc_train:
            best_acc_train = epoch_acc
        if phase == "train" and epoch_loss < best_loss_train:
            best_loss_train = epoch_loss

# Print final results
model.load_state_dict(best_model_wts)
time_elapsed = time.time() - t0
print(
    "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
)
print(f"Best validation loss: {best_loss:.4f} | Best validation accuracy: {best_acc:.4f}")

print("End time at:", datetime.datetime.now())
