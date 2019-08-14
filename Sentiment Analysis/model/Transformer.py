import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_pos_onehot(BATCH_SIZE, average_len):
    embeddedPosition = []
    for batch in range(BATCH_SIZE):
        x = []
        for step in range(average_len):
            a = np.zeros(average_len)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)
   
    return torch.tensor(embeddedPosition, dtype=torch.float)

class TransformerModel(nn.Module):

    def __init__(self,vocab_size,average_len,batch_size,embedding_dim,pre_trained_enbedding,model_size,num_heads,num_blocks,dropout,output_size):
        super(TransformerModel,self).__init__()
        #一些要用的参数
        self.model_size = model_size
        self.batch_size = batch_size
        self.seq_len = average_len

        #嵌入层（使用预训练的词向量）
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(pre_trained_enbedding)
        self.word_embeddings.weight.requires_grad = False

        #将嵌入层的维度经过线性变换成模型维度（300 -- 64/128/256/512）。位置信息做同样的操作。
        self.emb_ff = nn.Linear(embedding_dim,model_size)
        self.pos_ff = nn.Linear(average_len,model_size)

        #进入Encoder中，传入参数分别为（）
        self.encoder = Encoder(model_size,model_size,num_blocks,num_heads,dropout)   
        
        #输出时，进入一个或两个线性层，将维度最终降到3维
        self.Linear = nn.Linear(model_size,64)
        self.Linear1 = nn.Linear(64,output_size)

    def forward(self,x):
        #论文里此处embeds乘了根号下model_size，暂且不乘，后期再看 #embeds = embeds*np.sqrt(self.model_size)
        embeds = self.word_embeddings(x)
        embeds = self.emb_ff(embeds)
        
        #position选择很多，可以用one-hot encoding，也可以用论文里的cos/sin值。此处选用one-hot。映射到128维然后与wordembedding相加。
        position_code = get_pos_onehot(embeds.size(0),self.seq_len).cuda()
        position_code = self.pos_ff(position_code)
        pos_embedding = (embeds + position_code)

        #进入encoder最终得到输出
        output = self.encoder(pos_embedding)
        #从encoder出来后，有几个词就会有几个向量，此处取所有词的均值
        output = output.mean(dim=1)
        output = self.Linear(output)
        output = self.Linear1(output)
        probs = F.log_softmax(output, dim=1)
        return probs

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_blocks,num_heads,activation=nn.ReLU,dropout=0.1):
        super(Encoder,self).__init__()
        #这里是一个循环，有几个block就循环几次，即做几次encoder，论文里是6次
        self.blocks = nn.Sequential(
            *[Block(input_size,hidden_size,num_heads,activation,dropout=dropout) 
                for i in range(num_blocks)]
                )

    def forward(self,x):
        #调用block
        output = self.blocks(x)
        return output

class Block(nn.Module):
    def __init__(self,input_size,hidden_size,num_heads,activation=nn.ReLU,dropout=0.1):
        super(Block,self).__init__()
        self.dropout = dropout   
        self.attention = MultiHeadAttention(input_size,hidden_size,num_heads)
        self.attention_norm = nn.LayerNorm(input_size)
        
        #这么写是为了，如果有dropout层，可以append进列表，进行dropout
        ff_layers = [
            nn.Linear(input_size,hidden_size*4),
            nn.ReLU(),
            nn.Linear(hidden_size*4,input_size),
            ]
        
        #这么写是为了，如果有dropout层，可以append进列表，进行dropout
        if self.dropout:
            self.attention_dropout = nn.Dropout(dropout)
            ff_layers.append(nn.Dropout(dropout))

        #先做一次线性层，加个激活函数relu，再做一次线性层。（为什么要进前馈神经网络，有待研究）
        self.ff = nn.Sequential(*ff_layers)
        #再进行一次归一化
        self.ff_norm = nn.LayerNorm(input_size)

    def forward(self,x):
        #正式进入encoder结构
        #先搞多头attention，三个相同的输入是因为要乘以不同的权重矩阵产生query，key，value
        mulit_attention = self.attention(x,x,x)
        #再搞归一化（此处没有残差，后面可以加上），上一阶段输出维度为【2，25，128】，现进入归一化。
        #此处用的是layer_norm，原理还欠研究。
        mulit_attention_after_norm = self.attention_norm(mulit_attention + x)
        #归一化之后，进入全连接（前馈神经网络）
        #维度仍为【2，25，128】
        feed_forword = self.ff(mulit_attention_after_norm)
        #再来一次归一化
        feed_forword_after_norm = self.ff_norm(feed_forword + x)
        #可以返回结果了，结果维度为【2，25，128
        return feed_forword_after_norm

class MultiHeadAttention(nn.Module):
    #其实hidden_size可以为64维，这样与论文中一致。
    def __init__(self,input_size,hidden_size,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.input_size = input_size#128
        self.hidden_size = hidden_size#128
        self.num_heads = num_heads#4
              
        #4个头，平分hidden_size
        self.head_size = self.hidden_size // num_heads
        #做线性映射，相当于乘以权重矩阵W_q,W_k,W_v, 这些参数其实为线性层神经元的参数
        self.q_linear = nn.Sequential(nn.Linear(self.input_size,self.hidden_size),nn.ReLU())
        self.k_linear = nn.Sequential(nn.Linear(self.input_size,self.hidden_size),nn.ReLU())
        self.v_linear = nn.Sequential(nn.Linear(self.input_size,self.hidden_size),nn.ReLU())
        
        self.softmax = nn.Softmax(dim=-1)
        self.joint_linear = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.Dropout(0.1))    

    def forward(self,q,k,v):
        #做线性映射，相当于乘以权重矩阵W_q,W_k,W_v, 这些参数其实为线性层神经元的参数
        query = self.q_linear(q)
        key = self.k_linear(k)
        value = self.v_linear(v)
        #原维度【batch，sent，embedding】 view成【batch，sent，头数，embedding平均到每个头】。此处就相当于把128维词向量拆分成4组，每组32维。
        #transpose（1，2）是把头数和句长换位置，这样原词向量就变短了，可以理解为一个句子（25个词）的词向量大小为【25X32】。总共有4个这样的词向量
        query = query.view(q.size(0), q.size(1), self.num_heads,self.head_size).transpose(1,2)
        key = key.view(q.size(0), q.size(1), self.num_heads,self.head_size).transpose(1,2)
        value = value.view(q.size(0), q.size(1), self.num_heads,self.head_size).transpose(1,2)
        
        #得到了query，key，value，可以计算相似度了，通过query和key点积得到，现在是矩阵，只能矩阵得到了。
        #所以对key进行转置成【32X25】，计算完后得到【25X25】相当于，25个词之间每两个词的分数。
        score = torch.matmul(query,key.transpose(2,3))
        #分数除以根号下Key的维度，即32，再经过softmax就得到了attention，即weight，每两个词之间的weight。
        weights = self.softmax(score / torch.sqrt(torch.Tensor([self.head_size * 1.0]).to(score)))
        
        # 用得到的权重乘以value（value代表当前词），得到每个词该给的真实value
        weighted_v = torch.matmul(weights,value)
        #再将得到的weight【2，4，25，32】变成【2，25，4，32】
        weighted_v = weighted_v.transpose(1,2).contiguous()
        
        # 【2，25，4，32】变成【2，25，128】相当于把四个头合并起来了，再经过一个线性层，相当于乘以权重矩阵。
        sum_value = self.joint_linear(weighted_v.view(q.size(0),q.size(1),self.hidden_size))
        #可以将结果送入前馈层了（在此之前要先归一化一波）
        return sum_value

