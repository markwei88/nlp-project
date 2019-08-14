import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class LSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pre_trained_enbedding, hidden_size, layer_num, bidirectional, attention_size, output_size):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirectional
        self.attention_size = attention_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(pre_trained_enbedding)
        self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_size,
                            self.layer_num,
                            batch_first = True,
                            bidirectional = self.bidirectional,
                            dropout = 0.25,
                           )
        self.w = torch.zeros(2*hidden_size, self.attention_size).cuda() if self.bidirectional else torch.zeros(hidden_size, self.attention_size).cuda()
        self.u = torch.zeros(self.attention_size).cuda() 
        self.fc1 = nn.Linear(self.hidden_size*2, 64) if self.bidirectional else nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64,output_size)

    def attention_net(self, lstm_output):
        if self.bidirectional:
            hidden_size = self.hidden_size * 2
        else:
            hidden_size = self.hidden_size
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, hidden_size])
        # print(output_reshape.size())
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w))
        # print(attn_tanh.size())
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u, [-1, 1]))
        # print(attn_hidden_layer.size())
        
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size(1)])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1]) 
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size(1), 1])
        # print(lstm_output.size())
        state = lstm_output.permute(0, 1, 2)
        # print(state.size())
        # print(alphas_reshape.size())
        attn_output = torch.sum(state * alphas_reshape, 1)
        
        return attn_output
        
    def forward(self,inputs):
        embeds = self.word_embeddings(inputs)
        h0 = torch.zeros(self.layer_num * 2, embeds.size(0), self.hidden_size).cuda() if self.bidirectional else torch.zeros(self.layer_num, embeds.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.layer_num * 2, embeds.size(0), self.hidden_size).cuda() if self.bidirectional else torch.zeros(self.layer_num, embeds.size(0), self.hidden_size).cuda()
        out, (hn, cn) = self.lstm(embeds, (h0, c0))
        attn_output = self.attention_net(out)
        out = self.fc1(attn_output)
        out = self.fc2(out)
        probs = F.log_softmax(out, dim=1)
        
        return probs

