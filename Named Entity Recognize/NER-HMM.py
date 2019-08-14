#==============================================================================
# Importing
from collections import Counter
from itertools import product
from sklearn.metrics import f1_score
import argparse
import random
import time 

#==============================================================================
# Command line processing
class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser()
        #define a Incompatible parameters group
        group = parser.add_mutually_exclusive_group()
        group.add_argument("-v",'--viterbi',help = 'run Viterbi algorithm',action="store_true")
        group.add_argument("-b",'--beam',help = 'run Beam Search algorithm',action="store_true")

        parser.add_argument("train_file",help='Get training data file')
        parser.add_argument("test_file",help='Get test data file')
        args= parser.parse_args()

        self.trainfile = args.train_file #Get the training file
        self.testfile=args.test_file #Get the question file
        self.viterbi = args.viterbi
        self.beam = args.beam

#==============================================================================
# Loading data set
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None): 
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()] 
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()] 
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

#==============================================================================
#The function is to get the counts of current word and current label
def cw_cl(train_data):
    #Create a list to store pairs of current word and current label
    word_pair_list = []
    for sentence in train_data:
        for word_pair in sentence:
            word_pair_list.append(word_pair)
    #Get the counts of cw_cl
    cw_cl_counts = dict(Counter(word_pair_list))
    return cw_cl_counts

#==============================================================================
#The function is to get the given sentence's counts of current word and label
def phi_1(words_list,tags_list,cw_cl_counts):
    pairs_list = []
    cw_cl_counts_given_sentence = {}

    for i in range(len(words_list)):
        pair = (words_list[i],tags_list[i])
        #Only append the pair in 'cw_cl_counts'
        if pair in cw_cl_counts: 
            pairs_list.append(pair)

    cw_cl_counts_given_sentence = dict(Counter(pairs_list))  

    return cw_cl_counts_given_sentence

#==============================================================================
#The function is to judge whether current word and current label in cw_cl_counts
def phi(word,tag,cw_cl_counts):
    pair = (word,tag)
    if pair in cw_cl_counts:
        result = 1
    else:
        result = 0

    return result

#==============================================================================
#The function is to use 'viterbi' algorithm to predict tag for given sentence
def viterbi_predict(weight,words_list,cw_cl_counts):
    y_hat = []
    #Define a list include 5 score (The initial value is 0)
    final_score_list = [0,0,0,0,0]
    tags_5 = ['O','PER','LOC','ORG','MISC']
    
    for i in range(len(words_list)):
        best_score_list = []
        #Get the current tag
        for current_tag in tags_5:
            score_list = []
            #Get pair from current word and current tag   
            pair = (words_list[i],current_tag)    
            #Get the previous tag
            for previous_tag in tags_5:
                score = 0
                #Calculate candidate scores for all possible
                score = score + final_score_list[i] + weight[pair]*phi(words_list[i],current_tag,cw_cl_counts) if pair in weight else score + final_score_list[i]
                score_list.append(score)
            #Find the best score for every current tag
            best_score_list.append(max(score_list))
        #Update previous tag's score
        final_score_list[i] = final_score_list[i] + best_score_list[i]
        #Get highest score from 5 tags as the most possible tag
        max_index = best_score_list.index(max(best_score_list))
        y_hat.append(tags_5[max_index])

    return y_hat

#==============================================================================
#The function is to use 'beam search' algorithm to predict tag for given sentence
def beam_predict(weight,words_list,cw_cl_counts,beam):
    y_hat = []
    tags_5 = ['O','PER','LOC','ORG','MISC']
    best_score_dic = {}
    
    for i in range(len(words_list)):
        #The first word's best score doesn't depend on previous tag
        if i == 0:
            for current_tag in tags_5:
                pair = (words_list[i],current_tag)
                score = weight[pair]*phi(words_list[i],current_tag,cw_cl_counts) if pair in weight else 0
                best_score_dic[current_tag] = score
            
            #Take the top beam size's scores and tags
            best_score_list = sorted(best_score_dic.items(),key = lambda x : x[1],reverse = True)
            best_score_list = best_score_list[:beam]
    
        else:
            #Create a dictionary to store new tags and their score
            best_score_dic = {}
            for current_tag in tags_5:
                for j in range(len(best_score_list)):
                    pair = (words_list[i],current_tag)
                    #Calculate best score for all possible combination
                    score = best_score_list[j][1] + weight[pair]*phi(words_list[i],current_tag,cw_cl_counts) if pair in weight else best_score_list[j][1]
                    #If it is the second word, the tag's type need to transform to tuple
                    if type(best_score_list[j][0]) != tuple:
                        new_key = (best_score_list[j][0],) + (current_tag,)
                    else:
                        new_key = best_score_list[j][0] + (current_tag,)
        
                    best_score_dic[new_key] = score

            #Take the top beam size's scores and tags
            best_score_list = sorted(best_score_dic.items(),key = lambda x : x[1],reverse = True)
            best_score_list = best_score_list[:beam]
    
    if type(best_score_list[0][0]) == str:
        y_hat.append(best_score_list[0][0])
    else:
        y_hat = list(best_score_list[0][0])

    return y_hat

#==============================================================================
#The function is to train weight 
def train(weight,train_data,cw_cl_counts,beam = None):
    #Randomly shuffle train data
    random.seed(666)
    random.shuffle(train_data)
    #Get words_list and tags_list from sentence
    for sentence in train_data:
        words_list = []
        tags_list = []
        for pair in sentence:
            words_list.append(pair[0])
            tags_list.append(pair[1])

        # 'y' is the true tags_list   
        y = tags_list
        #Get phi1 for updating weight
        y_phi = phi_1(words_list,tags_list,cw_cl_counts)
        #Get predict tags by calling predict
        if beam == None:
            y_hat = viterbi_predict(weight,words_list,cw_cl_counts)
        else:
            y_hat = beam_predict(weight,words_list,cw_cl_counts,beam)

        y_hat_phi = phi_1(words_list,y_hat,cw_cl_counts)
        
        #After getting the predict tags and true tags, judge whether they are equal or not
        #If not, update the weight 
        if y != y_hat:
            new_phi = Counter(y_phi)
            new_phi.subtract(y_hat_phi)
            new_phi = dict(new_phi)
            for key,value in new_phi.items():
                if key in weight:
                    weight[key] = weight[key] + value
                else:
                    weight[key] = value                    
        else:
            continue
            
    return weight

#==============================================================================
# This function is to iterate the weight 
def average_weight(train_data,iteration,beam = None):
    cw_cl_counts = cw_cl(train_data)
    weight = {}
    weight_total = {}
    all_weight = []

    # Iterate the weight   
    for i in range(iteration):
        weight = train(weight,train_data,cw_cl_counts,beam)
        # Everytime store weight to the list for later use
        all_weight.append(weight)

    #Sum all weight's value    
    for w in all_weight:
        for key,value in w.items():
            if key not in weight_total:
                weight_total[key] = 0
                weight_total[key] = weight_total[key] + value
            else:
                weight_total[key] = weight_total[key] + value

    #Get the average weight 
    for key,value in weight_total.items():
        weight_total[key] = value/iteration
        
    return weight_total 

#==============================================================================
# This function is to get the predict tags for test data 
# The parameter 'all_words_list' is words of all sentence in test data
def get_y_predict(weight,all_words_list,cw_cl_counts,beam = None):
    y_predict = []
    for words_list in all_words_list:
        if beam == None:
            y_hat = viterbi_predict(weight,words_list,cw_cl_counts)
        else:
            y_hat = beam_predict(weight,words_list,cw_cl_counts,beam)
        for tags in y_hat:
            y_predict.append(tags)

    return y_predict

#==============================================================================
# MAIN
if __name__ == '__main__':
    config = CommandLine()

    #Get the data
    train_data = load_dataset_sents(config.trainfile)
    test_data = load_dataset_sents(config.testfile)
    #To get y_true and all words of all sentence in test data for getting predict y
    y_true = []
    words_list = []
    all_words_list = []
    cw_cl_counts = cw_cl(test_data)

    for sentence in test_data:
        for pair in sentence:
            words_list.append(pair[0])
            y_true.append(pair[1])

        all_words_list.append(words_list)
        words_list = []

    if config.viterbi:
        iteration_list = [1,5,10]
        total_time = 0
        for iteration in iteration_list:
            start = time.clock()
            # Train training data get the weight
            weight = average_weight(train_data, iteration)
            #Get the predict y 
            y_predict = get_y_predict(weight, all_words_list, cw_cl_counts)
            #Calculate the F1 score 
            f1_micro = f1_score(y_true, y_predict, average='micro', labels=['O','ORG', 'MISC', 'PER', 'LOC'])
            #Print the result "F1 score and top 10 positive features"
            elapsed = (time.clock() - start)
            total_time = total_time + elapsed
            print('When using viterbi algorithm and the iteration times = %d  : '%iteration)
            print('The F1 score = :',f1_micro)
            print("Time used %f seconds:" %elapsed)
            print(' ')
        average_time = total_time/sum(iteration_list)
        print('The average time is : %f seconds' %average_time)
   
    if config.beam:
        iteration_list = [1,5,10]
        beam_list = [1,5,10]
        for beam in beam_list:
            #In order to ensure when beam size changing the train_data and test_data are same 
            #Reread the data here
            total_time = 0
            train_data = load_dataset_sents(config.trainfile)
            test_data = load_dataset_sents(config.testfile)
            for iteration in iteration_list:
                start = time.clock()
                # Train training data get the weight
                weight = average_weight(train_data, iteration, beam)
                #Get the predict y 
                y_predict = get_y_predict(weight, all_words_list, cw_cl_counts,beam)
                #Calculate the F1 score 
                f1_micro = f1_score(y_true, y_predict, average='micro', labels=['O','ORG', 'MISC', 'PER', 'LOC'])
                #Print the result "F1 score and top 10 positive features"
                elapsed = (time.clock() - start)
                total_time = total_time + elapsed
                print('When the beam size = %d' %beam)
                print('When using beam search algorithm and the iteration times = %d  : '%iteration)
                print('The F1 score = :',f1_micro)
                print("Time used %f seconds:" %elapsed)
                print(' ')
            average_time = total_time/sum(iteration_list)
            print('The average time is : %f seconds' %average_time)
            print(' ')

