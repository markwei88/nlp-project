import argparse
import torch
import csv
import os
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data
from evaluation import evaluate
from model.TextCNN import CNN
from model.LSTM import LSTM
from model.LSTM_Attention import LSTM_Attention
from model.Transformer import TransformerModel
from pre_process import pre_process
from create_dataloader import prepare_data
from load_wordembedding import load_word_embedding
from sklearn.model_selection import train_test_split

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--word_embedding_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Word Embedding Path")
    parser.add_argument("--model", 
                        default='CNN', 
                        type=str, 
                        required=True,
                        help="CNN/LSTM/LSTM+Attention")
    parser.add_argument("--output_dir", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="The result path")
    parser.add_argument("--output_name", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="The result path")
    parser.add_argument("--max_length",
                        default=25,
                        type=int,
                        help="Maximum sequence length, sequences longer than this are truncated")
    parser.add_argument("--epochs", 
                        default=15, 
                        type=int, 
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", 
                        default=0.001, 
                        type=float, 
                        dest="learning_rate",
                        help="Learning rate for optimizer")
    parser.add_argument("--device", 
                        default="cuda:0", 
                        dest="device",
                        help="Device to use for training and evaluation e.g. (cpu, cuda:0)")
    parser.add_argument("--dropout", 
                        default=0.1, 
                        type=float, 
                        dest="dropout",
                        help="Dropout (not keep_prob, but probability of ZEROING during training, i.e. keep_prob = 1 - dropout)")
    parser.add_argument("--batch_size", 
                        default=64, 
                        type=int, 
                        help="Batch size")
    parser.add_argument("--filter_sizes", 
                        default=[1,2,3,4,5], 
                        type=list, 
                        help="The filter sizes(CNN model)")
    parser.add_argument("--num_filters", 
                        default=50, 
                        type=int, 
                        help="The number of filters(CNN model)")
    parser.add_argument("--hidden_size", 
                        default=64, 
                        type=int, 
                        help="The number of hidden_size(LSTM/LSTM_Attention model)")
    parser.add_argument("--layer_num", 
                        default=1, 
                        type=int, 
                        help="The number of layer_num(LSTM/LSTM_Attention model)")
    parser.add_argument("--bidirectional", 
                        default=True, 
                        type=bool, 
                        help="Is that bidirectional or not(LSTM/LSTM_Attention model)")
    parser.add_argument("--attention_size", 
                        default=32, 
                        type=int, 
                        help="The dim of attention(LSTM_Attention model)")
    parser.add_argument("--model_size", 
                        default=128, 
                        type=int, 
                        help="The size of transformer's model(Transformer model)")
    parser.add_argument("--num_heads", 
                        default=4, 
                        type=int, 
                        help="The number of heads(Transformer model)")
    parser.add_argument("--num_blocks", 
                        default=2, 
                        type=int, 
                        help="The number of block(Transformer model)")

    args = parser.parse_args()

    print('......................Loading Data......................')
    x_trainval,y_trainval,x_test,y_test = get_data(args.data_dir)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=66)
    word_embedding = load_word_embedding(args.word_embedding_path)
    x_train,x_val,x_test,average_len,vocab = pre_process(x_train,x_val,x_test)
    vocab = set(vocab)


    #Create the dictionary word_to_index and target_to_index 
    word_to_id = {word : index for index,word in enumerate(vocab)}
    target_to_id = {target : index for index,target in enumerate(set(y_trainval))}
    id_to_target = {value: key for key, value in target_to_id.items()}

    #Define some hyperparameters
    embedding_dim = 300
    vocab_size = len(vocab) 
    output_size = 3
    pre_trained_enbedding = torch.zeros(vocab_size, embedding_dim)

    for key,value in word_to_id.items():
        if key in word_embedding and (key != '<pad>'):
            pre_trained_enbedding[value,:] = torch.from_numpy(word_embedding[key])

    #Transform data from text to tensor and put them in the Dataloader(for batch)
    train_loader = prepare_data(x_train,y_train,average_len,word_to_id,target_to_id,vocab,args.batch_size)
    val_loader = prepare_data(x_val,y_val,average_len,word_to_id,target_to_id,vocab,args.batch_size)
    test_loader = prepare_data(x_test,y_test,average_len,word_to_id,target_to_id,vocab,args.batch_size)

    #Build Model
    if args.model == 'CNN':
        model = CNN(vocab_size,embedding_dim,pre_trained_enbedding,args.filter_sizes, args.num_filters,args.dropout,output_size).to(args.device)
    if args.model == 'LSTM':
        model = LSTM(vocab_size,embedding_dim,pre_trained_enbedding,args.hidden_size,args.layer_num,args.bidirectional,output_size).to(args.device)
    if args.model == 'LSTM_Attention':
        model = LSTM_Attention(vocab_size, embedding_dim, pre_trained_enbedding, args.hidden_size, args.layer_num, args.bidirectional, args.attention_size, output_size).to(args.device)
    if args.model == 'Transformer':
        model = TransformerModel(vocab_size,average_len,args.batch_size,embedding_dim,pre_trained_enbedding,args.model_size,args.num_heads,args.num_blocks,args.dropout,output_size).to(args.device)
    
    print(model)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    #Training Data
    print('......................Training Data......................')
    print('          ')
    losses = []
    best_recall_score = 0.0

    with open(os.path.join(args.output_dir, args.model + args.output_name + '_training_result.csv'),'w') as csvfile:
        fieldnames = ['Epoch','Loss','train_accuracy_score','train_recall_score','train_f1_score','val_accuracy_score','val_recall_score','val_f1_score','test_accuracy_score','test_recall_score','test_f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(args.epochs):
            total_loss = 0
            for batch_x,batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_x,batch_y = batch_x.cuda(),batch_y.cuda()           
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_function(logits, torch.max(batch_y, 1)[0])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss)
            
            # print('......................Epoch: %d, Loss: %f......................' %(epoch,total_loss))
            # print('......................Training Data Performance......................')
            train_accuracy_score,train_recall_score,train_f1_score = evaluate(train_loader,model,id_to_target)
            # print('......................Validation Data Performance......................')
            val_accuracy_score,val_recall_score,val_f1_score = evaluate(val_loader,model,id_to_target)


            if val_recall_score > best_recall_score:
                best_recall_score = val_accuracy_score
                # real_model = model.module
                torch.save(model.state_dict(), args.model+'2model_best.pth')
            # print('......................Test Data Performance......................')
            test_accuracy_score,test_recall_score,test_f1_score = evaluate(test_loader,model,id_to_target)
            writer.writerow({'Epoch': epoch, 
                'Loss': total_loss, 
                'train_accuracy_score' : train_accuracy_score,
                'train_recall_score': train_recall_score,
                'train_f1_score':train_f1_score, 
                'val_accuracy_score':val_accuracy_score, 
                'val_recall_score':val_recall_score,
                'val_f1_score':val_f1_score,
                'test_accuracy_score':test_accuracy_score,
                'test_recall_score':test_recall_score,
                'test_f1_score':test_f1_score})
            print('              ')


    if args.model == 'CNN':
        model = CNN(vocab_size,embedding_dim,pre_trained_enbedding,args.filter_sizes, args.num_filters,args.dropout,output_size)
    if args.model == 'LSTM':
        model = LSTM(vocab_size,embedding_dim,pre_trained_enbedding,args.hidden_size,args.layer_num,args.bidirectional,output_size)
    if args.model == 'LSTM_Attention':
        model = LSTM_Attention(vocab_size,embedding_dim,pre_trained_enbedding,args.hidden_size,args.layer_num,args.bidirectional,args.attention_size,output_size)
    if args.model == 'Transformer':
        model = TransformerModel(vocab_size,average_len,args.batch_size,embedding_dim,pre_trained_enbedding,args.model_size,args.num_heads,args.num_blocks,args.dropout,output_size)

    checkpoint = torch.load(args.model+'2model_best.pth')
    model.load_state_dict(checkpoint)
    model.to(args.device)

    print('......................Test Data Performance......................')
    test_accuracy_score,test_recall_score,test_f1_score = evaluate(test_loader,model,id_to_target)
    print('The accuracy is:%f ' %test_accuracy_score)
    print('The macro_recall is:%f ' %test_recall_score)
    print('The macro_F_score is:%f ' %test_f1_score)
    print('              ')

if __name__ == "__main__":
    main()




# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model CNN --output_dir .\result --output_name a --num_filters 50 
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model CNN --output_dir .\result --output_name b --num_filters 100 
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model CNN --output_dir .\result --output_name c --num_filters 50 --dropout 0.3

# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM --output_dir .\result --output_name d --hidden_size 32 --layer_num 1 --bidirection True
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM --output_dir .\result --output_name e --hidden_size 128 --layer_num 1 --bidirection True
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM --output_dir .\result --output_name f --hidden_size 64 --layer_num 2 --bidirection True
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM --output_dir .\result --output_name g --hidden_size 64 --layer_num 4 --bidirection True

# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM_Attention --output_dir .\result --output_name h --hidden_size 32
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM_Attention --output_dir .\result --output_name i --hidden_size 128
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM_Attention --output_dir .\result --output_name j --attention_size 64
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model LSTM_Attention --output_dir .\result --output_name k --attention_size 128

# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model Transformer --output_dir .\result --output_name q --model_size 256
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model Transformer --output_dir .\result --output_name w --model_size 512
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model Transformer --output_dir .\result --output_name e --num_heads 8 --num_blocks 4
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model Transformer --output_dir .\result --output_name r --model_size 512 --num_heads 8 --num_blocks 4 --ff*4
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model Transformer --output_dir .\result --output_name t --model_size 256 --num_heads 8 --num_blocks 4 --ff*4
# python .\train.py --data_dir .\data --word_embedding_path .\datastories.twitter.300d.txt --model Transformer --output_dir .\result --output_name y --model_size 128 --num_heads 8 --num_blocks 4 --ff*4