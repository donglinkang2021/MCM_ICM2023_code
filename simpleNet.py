import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        # x = F.relu(self.fc1(x))
        # x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x) # 将数据转换为概率分布
        return x
    
def word2vec(word):
    # 将单词转换为向量
    # 这里由于单词长度为5，所以将单词转换为长度为5的向量是很简单的
    # 而在自然语言处理中，单词的长度是不固定的，所以需要将单词转换为固定长度的向量
    # 还要做embedding
    vec = np.zeros(5)
    for i in range(len(word)):
        vec[i] = ord(word[i]) - ord('a')
    return vec

def vec2word(vec):
    # 将向量转换为单词
    word = ''
    for i in range(len(vec)):
        word += chr(int(vec[i]) + ord('a'))
    return word