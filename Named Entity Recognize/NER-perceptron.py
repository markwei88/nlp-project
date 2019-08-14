#==============================================================================
# Importing
import argparse
import copy
import random
from random import choice
from itertools import product
from collections import Counter
from sklearn.metrics import f1_score

#==============================================================================
# Command line processing
class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("train_file")
        parser.add_argument("test_file")
        args= parser.parse_args()

        self.trainfile = args.train_file #Get the training file
        self.testfile=args.test_file #Get the question file

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
#The function is to get the counts of previous label and current label
def pl_cl(train_data):
    #In order to avoid change train data, copy a new list for adding 'None'
    train_data_with_none = copy.deepcopy(train_data)
    #Create a new list to store previous and current label
    pre_cur_label_list = []
    
    for sentence in train_data_with_none:
        #Insert a tuple (None, None) to the first index of sentence
        sentence.insert(0,('None','None'))
        #Create new label consisting of previous and current label
        for i in range(len(sentence)-1):
            previous_current_label = (sentence[i][1],sentence[i+1][1])
            pre_cur_label_list.append(previous_current_label)
   
    #Get the counts of pl_cl
    pl_cl_counts = dict(Counter(pre_cur_label_list))

    return pl_cl_counts

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
#The function is to get the given sentence's counts of previous and current labels
def phi_2(words_list,tags_list,pl_cl_counts):
    pl_cl_count_list = []
    #In order to avoid change train data, copy a new list for adding 'None'
    words_list_withnone = copy.deepcopy(words_list) 
    tags_list_withnone = copy.deepcopy(tags_list) 
    #Insert a  'None' to the first index of words_list
    words_list_withnone.insert(0,'None')
    #Insert a  'None' to the first index of tags_list
    tags_list_withnone.insert(0,'None')
    
    #Create new label consisting of previous and current label
    for i in range(len(tags_list_withnone)-1):
        previous_current_label = (tags_list_withnone[i],tags_list_withnone[i+1])
        #Ensure the p_c_lable in the pl_cl_counts
        if previous_current_label in pl_cl_counts:
            pl_cl_count_list.append(previous_current_label)

    #Get the counts of every previous_current_label in given sentence
    previous_current_label = dict(Counter(pl_cl_count_list))
    
    return previous_current_label

#==============================================================================
# The function is to predict the tags for given sentence
# The input 'feature' is to determine the feature is phi1 or phi1+phi2
# This function doesn't have parameter 'tags_list', because it is useless(every sentence just needs words which is enough)
# Because I combine two features in one function, so EVERY TIME CALL this predict it must pass both cw_cl_counts and pl_cl_counts

def predict(weight, words_list, cw_cl_counts, pl_cl_counts, feature):
    #The tags_5 including all tags 
    tags_5 = ['O','PER','LOC','ORG','MISC']
    #According to the length of words_list to get all possible tags by using <product>
    words_length = len(words_list)
    all_possible_tags = list(product(tags_5, repeat= words_length))
    #Create a score list to store every possible's score
    score_list = []
    
    #Calculate all possible's score and put them in a list
    for tags in all_possible_tags:
        score = 0
        tags_list = list(tags)

        #If the fearure is only phi_1
        if feature == 1:
            #Get the candidate pair's counts from phi_1
            candidate_pairs_phi_1 = phi_1(words_list,tags_list,cw_cl_counts)
            #Calculate the score according to the wight and counts
            for key,value in candidate_pairs_phi_1.items():
                if key in weight:
                    score += weight[key]*value
                else:
                    score += 0
            score_list.append(score)

        #If the feature is phi1+phi2    
        else:
        	#Get the candidate pair's counts from phi_1
            candidate_pairs_phi_1 = phi_1(words_list,tags_list,cw_cl_counts)
            #Get the previous label and current label's counts from phi_2
            candidate_pairs_phi_2 = phi_2(words_list,tags_list,pl_cl_counts)
            #Merge the phi1 and phi2
            candidate_pair_Merged=candidate_pairs_phi_1.copy()
            candidate_pair_Merged.update(candidate_pairs_phi_2)
            
            #Calculate the score according to the wight and counts
            for key,value in candidate_pair_Merged.items():
                if key in weight:
                    score += weight[key]*value
                else:
                    score += 0
            score_list.append(score)
    
    #Get the maximum from score list        
    max_score = max(score_list)
    #Get all index of max score (Sometimes there are more than one)
    max_index_list = [i for i, j in enumerate(score_list) if j == max_score]
    #Randomly choose one from max_index_list(When there are more than one maximum)
    max_index = choice(max_index_list)
    #From all possible tags get the one that has max score(because they have the same index)
    max_possible_tags = all_possible_tags[max_index]
        
    return max_possible_tags

#==============================================================================
# The function is to train weight (i.e. Perceptron)
# The input 'feature' is to determine the feature is phi1 or phi1+phi2
def train(weight,train_data,cw_cl_counts,pl_cl_counts,feature):  
	#Randomly shuffle train data
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
        # If the feature only is phi1
        if feature == 1:
        	#Get phi1 for updating weight
            y_phi = phi_1(words_list, tags_list, cw_cl_counts)
            #Get predict tags by calling predict
            y_hat = predict(weight, words_list, cw_cl_counts, pl_cl_counts, feature)
            y_hat_phi = phi_1(words_list, y_hat, cw_cl_counts)
        #If the features are phi1 + phi2
        else :
        	#Get phi1 for updating weight
            y_phi_1 = phi_1(words_list,tags_list,cw_cl_counts)
            #Get phi1 for updating weight
            y_phi_2 = phi_2(words_list,tags_list,pl_cl_counts)
            #Merge the phi1 and phi2 to y_phi(i.e combine two true features to one)
            y_phi = y_phi_1.copy()
            y_phi.update(y_phi_2)

            #Get the predict tags by calling predict
            y_hat = predict(weight, words_list, cw_cl_counts, pl_cl_counts, feature)
            #Use predict tags to get phi1
            y_hat_phi_1 = phi_1(words_list,y_hat,cw_cl_counts)
            #Use predict tags to get phi1
            y_hat_phi_2 = phi_2(words_list,list(y_hat),pl_cl_counts)
            #Merge the y_hat_phi1 and y_hat_phi2 to y_hat_phi(i.e combine two true features to one)
            y_hat_phi = y_hat_phi_1.copy()
            y_hat_phi.update(y_hat_phi_2)

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
        #If yes, continue                  
        else:
            continue

    return weight

#==============================================================================
# This function is to iterate the weight 
def average_weight(train_data,iteration,feature):
    cw_cl_counts = cw_cl(train_data)
    pl_cl_counts = pl_cl(train_data)
    weight = {}
    weight_total = {}
    all_weight = []
    
    # Iterate the weight  
    for i in range(iteration):
        weight = train(weight, train_data, cw_cl_counts, pl_cl_counts, feature)
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
def get_y_predict(weight,all_words_list,cw_cl_counts,pl_cl_counts,feature):
    y_predict = []
    for words_list in all_words_list:
    	#Judge what is the feature 
        if feature == 1:
            y_hat = predict(weight,words_list,cw_cl_counts,pl_cl_counts,feature)
        else:
            y_hat = predict(weight,words_list,cw_cl_counts,pl_cl_counts,feature)

        for tags in y_hat:
            y_predict.append(tags)

    return y_predict

#==============================================================================
# MAIN
if __name__ == '__main__':
    config = CommandLine()

    #Get the structed data 
    train_data = load_dataset_sents(config.trainfile)
    test_data = load_dataset_sents(config.testfile)

    # Feature = 1 is only use one feature (i.e. phi_1)
    # Train training data get the weight 
    weight_1 = average_weight(train_data, iteration = 10, feature = 1)
    # Feature = 2 is combine two features (i.e. phi_1 and phi_2)
    # Train training data get the weight 
    weight_2 = average_weight(train_data, iteration = 10, feature = 2)

    #To get y_true and all words of all sentence in test data for getting predict y
    y_true = []
    words_list = []
    all_words_list = []
    cw_cl_counts = cw_cl(test_data)
    pl_cl_counts = pl_cl(test_data)

    for sentence in test_data:
        for pair in sentence:
            words_list.append(pair[0])
            y_true.append(pair[1])

        all_words_list.append(words_list)
        words_list = []

    #Get the predict y only use phi1
    y_predict_phi1 = get_y_predict(weight_1, all_words_list, cw_cl_counts, pl_cl_counts, feature = 1)
    #Get the predict y use phi1 and phi2
    y_predict_phi2 = get_y_predict(weight_2, all_words_list, cw_cl_counts, pl_cl_counts, feature = 2)

    #Calculate the F1 score (phi1)
    f1_micro_phi1 = f1_score(y_true, y_predict_phi1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    #Calculate the F1 score (phi1 + phi2)
    f1_micro_phi2 = f1_score(y_true, y_predict_phi2, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])

    #Sort the weight
    weight_1 = sorted(weight_1.items(),key = lambda x:x[1],reverse = True)
    weight_2 = sorted(weight_2.items(),key = lambda x:x[1],reverse = True)

    #Print the result "F1 score and top 10 positive features"
    print('When the iteration times is 10 and only using "current word and current" label as feature :')
    print('The F1 score is :',f1_micro_phi1)
    for i in range(10):
        print('The top %d positive features of phi1 is:'%(i+1),weight_1[i][0],'and the value is %d'%weight_1[i][1])
    print(' ')
    print('When the iteration times is 10 and using "current word and current label" and "previous label and current label" as features :')
    print('The F1 score is :',f1_micro_phi2)
    for i in range(10):
        print('The top %d positive features of phi1+phi2 is:'%(i+1),weight_2[i][0],'and the value is %d'%weight_2[i][1])



