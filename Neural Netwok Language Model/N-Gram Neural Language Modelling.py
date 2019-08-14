#==============================================================================
# Importing
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(666)

#==============================================================================
#Get the training sentence and split it
CONTEXT_SIZE = 2
train_sentence = """– The mathematician ran .
– The mathematician ran to the store .
– The physicist ran to the store .
– The philosopher thought about it .
– The mathematician solved the open problem .""".split()

#Get training trigrams
train_trigrams = [([train_sentence[i], train_sentence[i + 1]], train_sentence[i + 2])
            for i in range(len(train_sentence) - 2)]

vocab = set(train_sentence)
#Get index of every word
word_to_ix = {word: i for i, word in enumerate(vocab)}

#==============================================================================
# Define LM class
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size,hidden_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return out,log_probs

#==============================================================================
#The function is to get the score of 'physicist' and 'philosopher' of whole sentence
def get_score(model,trigrams,word_to_ix,target_list):
    i = 0
    total_score = 1
    for context,target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        #Get target word's score
        score,log_probs = model(context_idxs)
        #Turn tensor into numpy
        np_score = score.detach().numpy()[0]
        #Get the target word
        target_word = target_list[i]
        #Get the target word's index
        target_index = word_to_ix[target_word]
        #Get the score of target word
        target_score = np_score[target_index]
        #Mutiply score for whole sentence
        total_score = total_score*target_score
        i += 1
    return total_score

#==============================================================================
torch.manual_seed(666)
losses = []
loss_function = nn.NLLLoss()
#Set different 5 hyper parameters
#The first parameter is embedding_dim
#The second parameter is epoch_times
#The third parameter is learning rate
#The last one is hidden_size
hyper_parameters_list = [[3,500,0.001,50],[5,500,0.01,50],[5,500,0.01,100],[8,800,0.001,100],[10,800,0.01,100]]
#Set three test sentences
test_sentence = '– The mathematician ran to the store .'.split()
test_sentence1 = '– The physicist solved the open problem .'.split()
test_sentence2 = '– The philosopher solved the open problem .'.split()
#Set target list for getting score for 'physicist' and 'philosopher'
target_list1 = ['physicist','solved','the','open','problem','.']
target_list2 = ['philosopher','solved','the','open','problem','.']

#Get test sentences' trigrams
test_trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
        for i in range(len(test_sentence) - 2)]
test_trigrams1 = [([test_sentence1[i], test_sentence1[i + 1]], test_sentence1[i + 2])
        for i in range(len(test_sentence1) - 2)]
test_trigrams2 = [([test_sentence2[i], test_sentence2[i + 1]], test_sentence2[i + 2])
        for i in range(len(test_sentence2) - 2)]

#Set standard answer for evaluating predictions of test_sentence
standard_answer = ['mathematician','ran','to','the','store','.']

#Traverse 5 groups‘ parameters
for hyper_parameters in hyper_parameters_list:
    #Get four parameters
    EMBEDDING_DIM = hyper_parameters[0]
    epoch_times = hyper_parameters[1]
    lr = hyper_parameters[2]
    hidden_size = hyper_parameters[3]

    #Learn model
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE,hidden_size)
    optimizer = optim.SGD(model.parameters(), lr)
    
    losses = []
    for epoch in range(epoch_times):
        total_loss = 0
        for context, target in train_trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            context_var = autograd.Variable(torch.LongTensor(context_idxs))
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()
            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            probs,log_probs = model(context_idxs)
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()

        losses.append(total_loss)
        # The loss decreased every iteration over the training data!
    #Plot the losses
    plt.figure()
    plt.plot(losses)
    
    #Run a Sanity check
    #Check whether this model work in correct answer
    predict_word_list = []
    for context,target in test_trigrams:
        #Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        #Get the prediction score
        score,log_probs = model(context_idxs)
        np_score = score.detach().numpy()[0]
        #Find the max value of score as prediction
        max_score = np.argwhere(np_score == np_score.max())
        #Get that word
        predict_word = list(word_to_ix.keys())[list(word_to_ix.values()).index(max_score)]
        predict_word_list.append(predict_word)
    
    print('**********************************************************')
    #Judge whether the predicted word is the same as the standard answer
    if predict_word_list == standard_answer: 
        print('When embedding_dim = %d, epoch_times = %d, learning_rate = %s, hidden_size = %d' %(EMBEDDING_DIM,epoch_times,lr,hidden_size))
        print('The prediction of words are :',predict_word_list)
        print('Correct prediction!')     
    else:
        print('Incorrect prediction!')
    print(' ')
    #Get the total scores of test sentence 1 and test sentence 2
    total_score_physicist = get_score(model,test_trigrams1,word_to_ix,target_list1)
    total_score_philosopher = get_score(model,test_trigrams2,word_to_ix,target_list2)
    #Get the higher score's sentence
    higher_score = 'physicist' if total_score_physicist > total_score_philosopher else 'philosopher'
    print('The score of physicist = %s'%total_score_physicist)
    print('The score of philosopher = %s'%total_score_philosopher)
    print('The higher score between physicist and philosopher is : %s' %higher_score)
    print(' ')
    
    #Get the embedded vector of three words ['mathematician','physicist','philosopher']
    context_idxs_mathematician = torch.tensor([word_to_ix['mathematician']], dtype=torch.long)
    context_idxs_physicist = torch.tensor([word_to_ix['physicist']], dtype=torch.long)
    context_idxs_philosopher = torch.tensor([word_to_ix['philosopher']], dtype=torch.long)
    tensor_mathematician = model.embeddings(context_idxs_mathematician)
    tensor_physicist = model.embeddings(context_idxs_physicist)
    tensor_philosopher = model.embeddings(context_idxs_philosopher)
    
    #Calculate the cosine similarity between ['mathematician' ,'physicist' ] and ['mathematician' ,'philosopher' ] 
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_mathematician_physicist = cos(tensor_mathematician, tensor_physicist).detach().numpy()[0]
    cos_mathematician_philosopher = cos(tensor_mathematician, tensor_philosopher).detach().numpy()[0]
    more_similar = 'mathematician and physicist' if cos_mathematician_physicist > cos_mathematician_philosopher else 'mathematician and philosopher'
    print('The cosine similarity between mathematician and physicist = %s' %cos_mathematician_physicist)
    print('The cosine similarity between mathematician and philosopher = %s' %cos_mathematician_philosopher)
    print('More similar two words are : %s'%more_similar)
    print(' ')
    
    