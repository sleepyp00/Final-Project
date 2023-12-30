from torch import nn
import torch 
import torch.nn.functional as F

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