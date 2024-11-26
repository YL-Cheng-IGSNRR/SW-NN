# 建立
import torch
import torch.nn as nn
import torch.nn.functional as F


class SW_NN(torch.nn.Module):
    def __init__(self, n_feature, n_coe, n_neuron, n_layer):  # n_feature:输入特征数目, n_neuron为中间层神经元数目, n_layer为隐藏层数目, n_coe为输出层神经元数目
        self.n_feature = n_feature
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.n_coe = n_coe
        super(SW_NN, self).__init__()
        
        self.relu = nn.ReLU()
        # input layer
        self.block1 = nn.Sequential(nn.Linear(self.n_feature, self.n_neuron), nn.BatchNorm1d(self.n_neuron), nn.ReLU())
        # [Linearm BN, ReLu] repeart n_layer times
        self.block_temp = nn.Sequential(nn.Linear(self.n_neuron, self.n_neuron), nn.BatchNorm1d(self.n_neuron), nn.ReLU())
        self.block2 = nn.Sequential(*[self.block_temp for i in range(self.n_layer)])
        # shortcut
        self.layer_shortcut = nn.Linear(self.n_feature, self.n_neuron)
        self.bn_shortcut = nn.BatchNorm1d(self.n_neuron)
        # output layer
        self.layer_3 = nn.Linear(self.n_neuron, self.n_coe)

    def forward(self, input_tensor):

        x = self.block1(input_tensor)

        x = self.block2(x)
        
        shortcut = self.layer_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut)
        x = x + shortcut
        x = self.relu(x)
        
        x = self.layer_3(x)

        return x
