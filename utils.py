import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve

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
            weight = torch.where(torch.argmax(output_probs, dim =1) == labels, 1.0, 0.0)
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

def train_model(model, epochs, loss_function, optimizer, scheduler, dataloaders, save_path = "Models/model.pth", attacker = None):
    model = model.to(device)
    
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

            if attacker is not None:
                images = attacker.attack(model, images, labels)

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
                    torch.save(model, save_path)
            model.train()
            
        if scheduler is not None:
            scheduler.step()

    if not perform_val:
        torch.save(model, save_path)
    model = torch.load(save_path)

def train_ensemble_standard(DE, epochs, loss_function, optimizer, scheduler, dataloaders, save_path = "Models/model.pth", save_each:bool = False, attacker = None):
    def get_sub_path(model_nr):
        return save_path[:save_path.rfind(".")] + "_" + str(model_nr) + ".pth"

    for i, model in enumerate(DE):
        print("Training sub model nr", i)
        if save_each:
            train_model(model, epochs, loss_function, optimizer, scheduler, dataloaders, save_path=get_sub_path(i), attacker = attacker)
        else:
            train_model(model, epochs, loss_function, optimizer, scheduler, dataloaders, attacker = attacker)
    torch.save(DE, save_path)

