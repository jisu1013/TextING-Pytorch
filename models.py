from layers import *
from parse import parse_args
import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, labels):

        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.args = parse_args()
        self.output_dim = labels.shape[1]
        self.layers = nn.ModuleList()
        self.labels = labels
        self.layers.append(Dense_layer(input_dim=self.input_dim,
                                        output_dim=self.args.hidden,
                                        act=F.relu, dropout=True, sparse_inputs=False))
        self.layers.append(Readout_layer(input_dim=self.args.hidden,
                                        output_dim=self.output_dim,
                                        act=lambda x: x, dropout=True))
    
    def _l2_loss(self, t):
        return torch.sum(t ** 2) / 2

    def forward(self, inputs, mask):

        x = self.layers[0](inputs)
        output = self.layers[1](x, mask)
        l2_loss = 0.
        # weight decay loss
        for weight in self.layers[0].weights:
            l2_loss += self.args.weight_decay * self._l2_loss(weight)
        bce_loss = F.binary_cross_entropy_with_logits(output, self.labels) 
        loss = l2_loss + bce_loss

        predict = F.softmax(output)

        return predict, loss


class GNN(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = parse_args()
        self.layers = nn.ModuleList()
        self.layers.append(Graph_layer(input_dim=self.input_dim, output_dim=self.args.hidden, dropout=self.args.dropout,
                                        act=F.tanh, sparse_inputs=False, steps=self.args.steps))
        self.layers.append(Readout_layer(input_dim=self.args.hidden, output_dim=self.output_dim, dropout=self.args.dropout,
                                        act=F.tanh, sparse_inputs=False))

    def _l2_loss(self):

        l2_loss = 0.
        # weight decay loss
        for weight in self.layers[0].weights:
            l2_loss += self.args.weight_decay * (torch.sum(weight ** 2) / 2)
        
        return l2_loss
    
    def bce_loss(self, outputs, labels):

        bce_loss = F.binary_cross_entropy_with_logits(outputs, labels)
        bce_loss = torch.mean(bce_loss)

        return bce_loss
    
    def predict(self, outputs):

        predicts = F.softmax(outputs)

        return predicts

    def forward(self, inputs, support, mask):
        
        x = self.layers[0](inputs, support, mask)
        outputs = self.layers[1](x, mask)
        
        return outputs