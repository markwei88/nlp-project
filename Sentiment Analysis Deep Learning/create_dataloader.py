import torch
import numpy as np
import torch.utils.data as Data

def sent2id(sent,word_to_id,vocab):
    return np.array([word_to_id[w] if w in vocab else word_to_id['<unknown>'] for w in sent])

def target2id(target,target_to_id):
    return torch.tensor([target_to_id[target]],dtype=torch.long)

def text_to_tensor(x,average_len,word_to_id,vocab):
    X = np.zeros((len(x), average_len))
    for i in range(len(X)):
        index = sent2id(x[i],word_to_id,vocab)
        X[i,:] = index
    return torch.tensor(X, dtype=torch.long)

def label_to_tensor(y,target_to_id):
    Y = [target2id(target,target_to_id) for target in y]
    Y = torch.tensor(Y).view(-1,1)
    return Y

def prepare_data(x,y,average_len,word_to_id,target_to_id,vocab,batch_size):
    X = text_to_tensor(x,average_len,word_to_id,vocab)
    Y = label_to_tensor(y,target_to_id)
    torch_dataset = Data.TensorDataset(X , Y)
    loader = Data.DataLoader(
                    dataset = torch_dataset,
                    batch_size = batch_size,
                    shuffle=True,
                    num_workers = 2
                    )
    return loader