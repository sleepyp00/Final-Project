import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from ensemble import DeepEnsemble
from pathlib import Path
from torch import nn


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