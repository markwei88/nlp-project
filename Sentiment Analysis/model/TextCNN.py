import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_trained_enbedding, filter_sizes, num_filters, dropout, output_size):
        super(CNN, self).__init__()
        # filter_sizes = [1,2,3,4,5]
        # num_filters = 20
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(pre_trained_enbedding)
        self.word_embeddings.weight.requires_grad = False
        
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embedding_dim)) for K in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, output_size)
        
    def forward(self,inputs):
        embeds = self.word_embeddings(inputs) 
        x = embeds.view(embeds.size(0), 1, embeds.size(1), embeds.size(2))
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        out = self.fc1(x)
        probs = F.log_softmax(out, dim=1)

        return probs