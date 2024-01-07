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
from pathlib import Path
from training_setup import Training_Setup
from finetune_net import Network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.05, stratify=targets)

X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
X_test = torch.tensor(X_test, device=device, dtype=torch.float32)

y_train = torch.tensor(y_train, device=device)
y_test = torch.tensor(y_test, device=device)

model = Network(768, 5, [512])
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
            dataloaders=dataloaders, 
            save_path="Models/base_classifier.pth")

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

project = hopsworks.login()
mr = project.get_model_registry()



replace_model(mr,
              model.cpu(),
              name="base_classifier",
              version=1,
              description="This model is finetuned on news data",
              metrics = {"accuracy":acc})



