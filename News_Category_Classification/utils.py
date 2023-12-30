import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve
import joblib
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(data, batch_size, shuffle = False):
    dataloaders = {}
    if "train" in data:
        train_dataloader = DataLoader(data["train"], batch_size = batch_size, shuffle = shuffle)
        dataloaders["train"] = train_dataloader
        data.pop("train")

    for key, value in data.items():
        dataloaders[key] = DataLoader(value, batch_size = batch_size, shuffle = False)

    return dataloaders

def ECE(model, dataloader, n_bins = 15):
    corrects = torch.zeros(n_bins)
    confidence_sums = torch.zeros(n_bins)
    N = 0.0
    
    for images, labels in dataloader:
        images = images.to(device)

        # Go back to cpu because histogram does not work with cuda
        output_probs = model.probabilities(images).cpu()

        confidences, _ = torch.max(output_probs, dim = 1)

        corrects += torch.histogram(
            confidences, 
            bins = n_bins, 
            range=(0.0, 1.0),
            weight = torch.where(torch.argmax(output_probs, dim =1) == labels.cpu(), 1.0, 0.0)
            )[0]
        confidence_sums += torch.histogram(
            confidences, 
            bins = n_bins, 
            range=(0.0, 1.0),
            weight = confidences
            )[0]
        
        N += len(labels)
        
    ece = torch.sum(torch.abs(corrects - confidence_sums))/N
    return ece.item()

def accuracy(model, dataloader):
    correct = torch.zeros(1, device=device)
    N = torch.zeros(1, device=device)
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output_probs = model.probabilities(images)
        correct += torch.sum(torch.argmax(output_probs, dim =1) == labels)
        N += len(labels)

    accuracy = correct/N
    return accuracy.item()


def train_model(model, epochs, training_setup, dataloaders, save_path = None):
    model = model.to(device)

    loss_function, optimizer, scheduler = training_setup.create_training_setup(model)
    
    if "validation" in dataloaders:
        perform_val = True
        valDataLoader = dataloaders["validation"]
    else:
        perform_val = False
        valDataLoader = None

    trainDataLoader = dataloaders["train"]

    best_accuracy = 0.0

    for epoch in tqdm(range(epochs)):
        for i, (images, labels) in enumerate(trainDataLoader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

        if perform_val:
            model.eval()
            with torch.no_grad():
                acc = accuracy(model, valDataLoader)
                if acc > best_accuracy:
                    best_accuracy = acc
                    print("New best validation accuracy: ", best_accuracy)
                    if save_path is not None:
                        torch.save(model, save_path)
            model.train()
            
        if scheduler is not None:
            scheduler.step()

    if not perform_val and save_path is not None:
        torch.save(model, save_path)
    if save_path is not None:
        model = torch.load(save_path)

def train_ensemble_standard(DE, epochs, training_setup, dataloaders, save_path = "Models/model.pth", save_each:bool = False):
    def get_sub_path(model_nr):
        return save_path[:save_path.rfind(".")] + "_" + str(model_nr) + ".pth"

    for i, model in enumerate(DE):
        print("Training sub model nr", i)
        if save_each:
            train_model(model, epochs, training_setup, dataloaders, save_path=get_sub_path(i))
        else:
            train_model(model, epochs, training_setup, dataloaders)
    torch.save(DE, save_path)


def replace_model(mr, 
                  model,
                  name:str, 
                  version:int, 
                  new_name:str = None,
                  description:str = "", 
                  metrics = None):
    # Specify the directory path
    if new_name is None:
        new_name = name
    model_dir = Path("temp/model_" + new_name)

    # Create the directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / (new_name + ".pkl"))

    try:
        old_model = mr.get_model(name= name,
                    version=version)
        old_model.delete()
        print("deleted old version",version,"of model")
    except:
        print("Unable to retrieve old model for replacement")

    hw_model = mr.python.create_model(
        name=new_name, 
        version=version,
        metrics = metrics,
        description=description
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    hw_model.save(model_dir)

