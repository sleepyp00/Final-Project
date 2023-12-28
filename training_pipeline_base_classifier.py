import numpy as np 
import pandas as pd
import hopsworks
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from utils import train_ensemble_standard, train_model, accuracy, get_dataloaders, ECE, replace_model
import joblib
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from ensemble import DeepEnsemble
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Training_Setup:
    def __init__(
            self,
            lr:float,
            weight_decay:float = 0,
            gamma:float = 0.8,
            milestones:list = None,
            weights = None,
            ) -> None:
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.milestones = milestones
        self.weights = weights


    def create_training_setup(self, model):
        loss_function = nn.CrossEntropyLoss(weight=self.weights)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.milestones is not None:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, self.gamma)
        else:
            scheduler = None
        return (loss_function, optimizer, scheduler)

class Network(nn.Module):
    def __init__(self,input_dim:int, output_dim:int, layer_widths:list = []) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.output_dim = output_dim

        

        if len(layer_widths) > 0:
            self.FC_initial = nn.Linear(input_dim, layer_widths[0])
            self.hidden_layers = self.prepare_hidden_layers(layer_widths)
            self.FC_final = nn.Linear(layer_widths[-1], output_dim)
        else:
            self.FC_initial = nn.Linear(input_dim, output_dim)
            self.hidden_layers = nn.Sequential()
            self.FC_final = nn.Sequential()
        

    def prepare_hidden_layers(self, layer_widths):
        hidden_layers = [nn.Sequential(nn.Linear(layer_widths[i], layer_widths[i+1]), nn.ReLU()) for i in range(len(layer_widths) - 1)]
        #hidden_layers.append(nn.ReLU())
        return nn.Sequential(*hidden_layers)

    def forward(self, x):
        out = F.relu(self.FC_initial(x))
        out = self.hidden_layers(out)
        out = self.FC_final(out)
        return out

    def probabilities(self, x):
        return F.softmax(self.forward(x), dim = -1)



project = hopsworks.login()
fs = project.get_feature_store()

news_fg = fs.get_feature_group(name="basedataset", version=1)
query = news_fg.select_all()

feature_view = fs.get_or_create_feature_view(name="basedataset_view",
                                  version=1,
                                  description="View news classification dataset",
                                  query=query)

try:
    feature_view.delete()
except:
    print("No old view to delete")

feature_view = fs.get_or_create_feature_view(name="basedataset_view",
                                  version=1,
                                  description="View news classification dataset",
                                  query=query)

data, _ = feature_view.training_data()
""" X_train, X_test, _, _ = feature_view.train_test_split(0.1)
data = pd.concat([X_train, X_test], axis = 0) """
features = np.array(data["embedding"].values.tolist())
targets = data["categoryidx"].values

unique_values, counts = np.unique(targets, return_counts=True)

weights = counts.sum()/(len(unique_values)*counts)
weights = torch.tensor(weights, device=device, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1, stratify=targets)

X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
X_test = torch.tensor(X_test, device=device, dtype=torch.float32)

y_train = torch.tensor(y_train, device=device)
y_test = torch.tensor(y_test, device=device)

datasets = {
    "train":TensorDataset(X_train, y_train),
    "validation":TensorDataset(X_test, y_test)
}

model = Network(768, 5, [512])
""" model = DeepEnsemble(models=[Network(768, 5, [512]),
                             Network(768, 5, [512]),
                             Network(768, 5, [512]),
                             Network(768, 5, [512])])
 """
""" model = DeepEnsemble(models=[Network(768, 5, [512]),
                             ]) """
model.to(device=device)
model.train()

training_setup = Training_Setup(
        lr = 1e-3,
        weight_decay=1e-4,
        gamma=0.1,
        milestones=[10, 25, 45, 55],
        weights=weights
)
epochs = 75
batch_size = 64

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

dataloaders = {
    "train":train_loader,
    "validation":test_loader
}

train_model(model=model, 
            epochs=epochs, 
            training_setup=training_setup,
            dataloaders=dataloaders)

""" train_ensemble_standard(
    DE=model, 
            epochs=epochs, 
            training_setup=training_setup,
            dataloaders=dataloaders
) """

model.eval()
with torch.no_grad():
    acc = accuracy(model, test_loader)
    ece = ECE(model, test_loader)
    predictions = model.probabilities(X_test.to(device))
    predictions = predictions.cpu()
    y_test = y_test.cpu()

predicted_labels = np.argmax(predictions.numpy(), axis=1)
true_labels = y_test.numpy()

# Create the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("accuracy:", acc)
print("ece:", ece)

# Print or use the confusion matrix as needed
print("Confusion Matrix:")
print(conf_matrix)

""" project = hopsworks.login()
mr = project.get_model_registry()



replace_model(mr,
              model,
              name="base_classifier",
              version=1,
              description="This model is finetuned on news data") """

pass

