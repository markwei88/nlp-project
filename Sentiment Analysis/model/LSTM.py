import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_trained_enbedding,hidden_size, layer_num, bidirectional, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirectional
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(pre_trained_enbedding)
        self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_size,
                            self.layer_num,
                            batch_first = True,
                            bidirectional = self.bidirectional,
                           )
        self.fc1 = nn.Linear(self.hidden_size*2, 64) if self.bidirectional else nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64,output_size)
        
    def forward(self,inputs):
        embeds = self.word_embeddings(inputs)
        h0 = torch.zeros(self.layer_num * 2, embeds.size(0), self.hidden_size).cuda() if self.bidirectional else torch.zeros(self.layer_num, embeds.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.layer_num * 2, embeds.size(0), self.hidden_size).cuda() if self.bidirectional else torch.zeros(self.layer_num, embeds.size(0), self.hidden_size).cuda()
        out, (hn, cn) = self.lstm(embeds, (h0, c0))
              
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        probs = F.log_softmax(out, dim=1)
        
        return probs