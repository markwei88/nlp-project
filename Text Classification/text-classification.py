#==============================================================================
# Importing
import os
import tarfile
import gzip
import random
import re
import argparse
from collections import Counter
import matplotlib.pyplot as plt

#==============================================================================
# Command line processing
class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("infile")
        args= parser.parse_args()

        self.file = args.infile
       
#==============================================================================
# Unzip File
def Unzip(file_name):
    file_name = file_name+'.tar.gz'
    #Because the command 'python3 lab1.py review_polarity' without file suffix, so add '.tar.gz' firstly.
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    tar = tarfile.open(f_name)
    names = tar.getnames()
    if os.path.isdir(f_name + "_files"):
        pass
    else:
        os.mkdir(f_name + "_files")
    for name in names:
        tar.extract(name, f_name + "_files/")
    tar.close()
    foldername = str(f_name + '_files')
    return foldername

#==============================================================================
# Get files name
def get_files(foldername):
    neg_filename_list = []
    pos_filename_list = []
    for root, dirs, files in os.walk(foldername):
    #Extract every files and path under 'review_polarity'
        if root.endswith('/neg'):
            neg_path = root
            for filename in os.listdir(root):
                neg_filename_list.append(filename) 
        if root.endswith('/pos'):
            pos_path = root
            for filename in os.listdir(root):
                pos_filename_list.append(filename)
        #If the path suffix is '/neg' or '/pos', put every files name under this path to a list.
    neg_filename_list.sort()
    pos_filename_list.sort()
    #Sort all files name
    return neg_filename_list,pos_filename_list,neg_path,pos_path
    #Return the path of negative and positive since it will be used later.

#==============================================================================
# Create TEST data and TRAIN data
def train_test(neg_filename_list,pos_filename_list):
    train_dictionary = {}
    test_dictionary = {}

    for i in range(800):
        train_dictionary[neg_filename_list[i]] = -1 
        train_dictionary[pos_filename_list[i]] = 1
    #Take the first 800 files as training data with label '1'(positive) or '-1'(negative)
        
    for i in range(800,len(neg_filename_list)):
        test_dictionary[neg_filename_list[i]] = -1 
        test_dictionary[pos_filename_list[i]] = 1
    #Take the rest of files as test data with label '1'(positive) or '-1'(negative)

    return train_dictionary,test_dictionary

#==============================================================================
# Perceptron Algorithm
def Perceptron(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path):
    weight = {}
    filename = list(train_dictionary.keys())
    #Take keys of training dictionary as a list(for next step)
    random.shuffle(filename)
    #Shuffle the training files
    count_correct = 0
    count_files = 0
    count_files_list = []
    correct_rate_list = []
    #Define the above variables for plotting the learning progress
    for i in filename:
    #Traverse training files name
        if i in neg_filename_list:
        #If the file is from negative folder
            context = ''
            with open(neg_path+'/'+i) as f:
                for line in f.readlines():
                    context += line
            word_list = re.sub("[^\w']"," ",context).split()

            if feature == 'bigrams':
            #If the feature is 'bigrams'
                n = 0
                bigram_list = []
                while n!=len(word_list)-1:
                    bigram = word_list[n]+str(' ')+word_list[n+1]
                    bigram_list.append(bigram)
                    n += 1
                #Every two words make up a new word
                word_list = bigram_list
            elif feature == 'trigrams':
            #If the feature is 'trigrams'
                n = 0
                trigram_list = []
                while n!=len(word_list)-2:
                    trigram = word_list[n]+str(' ')+word_list[n+1]+str(' ')+word_list[n+2]
                    trigram_list.append(trigram)
                    n += 1
                #Every three words make up a new word
                word_list = trigram_list

            bag_of_words = Counter(word_list)
            #Get the every words' frequence by using Counter 

            score = 0
            for word in bag_of_words:
                if word not in weight:
                    weight[word] = 0
                #If this is the first time the word appear, add this word in weight dictionary and give weight 0
                score += weight[word] * bag_of_words[word]  
                #Conpute the score of this file
            if score >= 0 :
                score = 1
            else:
                score = -1 
            #If it is greater or equal to 0, then give it 1
            #If not, give it -1.

            if score != train_dictionary[i]:
                for word in bag_of_words:
                    weight[word] = weight[word] + train_dictionary[i] * bag_of_words[word]
            #If the prediction is different from the label, then update the weight
            else:
                count_correct += 1
            #If same, increase the correct number by one

        else:
            context = ''
            with open(pos_path+'/'+i) as f:
                for line in f.readlines():
                    context += line
            word_list = re.sub("[^\w']"," ",context).split()

            if feature == 'bigrams':
                n = 0
                bigram_list = []
                while n!=len(word_list)-1:
                    bigram = word_list[n]+str(' ')+word_list[n+1]
                    bigram_list.append(bigram)
                    n += 1
                word_list = bigram_list
            elif feature == 'trigrams':
                n = 0
                trigram_list = []
                while n!=len(word_list)-2:
                    trigram = word_list[n]+str(' ')+word_list[n+1]+str(' ')+word_list[n+2]
                    trigram_list.append(trigram)
                    n += 1
                word_list = trigram_list

            bag_of_words = Counter(word_list)

            score = 0
            for word in bag_of_words:
                if word not in weight:
                    weight[word] = 0
                score += weight[word] * bag_of_words[word]  
            if score >= 0 :
                score = 1
            else:
                score = -1 

            if score != train_dictionary[i]:
                for word in bag_of_words:
                    weight[word] = weight[word] + train_dictionary[i] * bag_of_words[word]
            else:
                count_correct += 1
        count_files += 1
        if count_files % 50 == 0:
            correct_rate = count_correct/count_files
            count_files_list.append(count_files)
            correct_rate_list.append(correct_rate)
            #Recode the correct rate every 50 files to plot the learning progress
    return weight,count_files_list,correct_rate_list

#==============================================================================
# Plot the learning progress
def plot_learning_progress(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path):
    weight,count_files_list,correct_rate_list = Perceptron(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path)
    if feature == 'unigrams':
        plt.plot(count_files_list,correct_rate_list)
        plt.xlabel('The number of files')
        plt.ylabel('The correct rate')
        plt.show()
    #I choose when feature is 'bigrams' to plot the learning progress

#==============================================================================
# Get the Average Weight
def average_weight(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path,iteration):
    weight_total = {}
    for c in range(iteration):
        weight,count_files_list,correct_rate_list = Perceptron(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path)
        for key,value in weight.items():
            if c == 0:
                weight_total[key] = value
            else:
                weight_total[key] = value + weight_total[key]

    for key,value in weight_total.items():
        weight_total[key] = value/iteration
    #Compute the average weight after many iterations

    return weight_total

#==============================================================================
# Top 10 positively-weighted features
def top_features(weight):
    sort_weight = sorted(weight.items(), key=lambda x:x[1],reverse = True)
    positive_features = sort_weight[0:10]
    negative_features = sort_weight[-10:]
    #After sorting the weight, return the top 10 negative and positive words and weight
    return positive_features,negative_features

#==============================================================================
# Using Test data to test model
def test_model(feature,test_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path,weight):
    filename = list(test_dictionary.keys())
    count = 0
    for i in filename:
        if i in neg_filename_list:
            context = ''
            with open(neg_path+'/'+i) as f:
                for line in f.readlines():
                    context += line
            word_list = re.sub("[^\w']"," ",context).split()

            if feature == 'bigrams':
                n = 0
                bigram_list = []
                while n!=len(word_list)-1:
                    bigram = word_list[n]+str(' ')+word_list[n+1]
                    bigram_list.append(bigram)
                    n += 1
                word_list = bigram_list
            elif feature == 'trigrams':
                n = 0
                trigram_list = []
                while n!=len(word_list)-2:
                    trigram = word_list[n]+str(' ')+word_list[n+1]+str(' ')+word_list[n+2]
                    trigram_list.append(trigram)
                    n += 1
                word_list = trigram_list

            bag_of_words = Counter(word_list)

            score = 0
            for word in bag_of_words:
                if word not in weight:
                    weight[word] = 0
                score += weight[word] * bag_of_words[word]  

            if score >= 0 :
                score = 1
            else:
                score = -1 

            if score == test_dictionary[i]:
                count += 1
            #If the prediction is same as the test label, increase the correct number by one

        else:
            context = ''
            with open(pos_path+'/'+i) as f:
                for line in f.readlines():
                    context += line
            word_list = re.sub("[^\w']"," ",context).split()

            if feature == 'bigrams':
                n = 0
                bigram_list = []
                while n!=len(word_list)-1:
                    bigram = word_list[n]+str(' ')+word_list[n+1]
                    bigram_list.append(bigram)
                    n += 1
                word_list = bigram_list
            elif feature == 'trigrams':
                n = 0
                trigram_list = []
                while n!=len(word_list)-2:
                    trigram = word_list[n]+str(' ')+word_list[n+1]+str(' ')+word_list[n+2]
                    trigram_list.append(trigram)
                    n += 1
                word_list = trigram_list

            bag_of_words = Counter(word_list)

            score = 0
            for word in bag_of_words:
                if word not in weight:
                    weight[word] = 0
                score += weight[word] * bag_of_words[word]  

            if score >= 0 :
                score = 1
            else:
                score = -1 

            if score == test_dictionary[i]:
                count += 1

    correct_rate = count/400
    #Use the number of correct classification files divide by 400 to get the correct rate
    return correct_rate

#==============================================================================
# MAIN
if __name__ == '__main__':
    
    config = CommandLine()
    foldername = Unzip(config.file)
    neg_filename_list,pos_filename_list,neg_path,pos_path = get_files(foldername)
    train_dictionary,test_dictionary = train_test(neg_filename_list,pos_filename_list)
    features_type = ['unigrams','bigrams','trigrams']
    for feature in features_type:
        weight = average_weight(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path,iteration=20)
        plot_learning_progress(feature,train_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path)
        positive_features,negative_features = top_features(weight)
        correct_rate = test_model(feature,test_dictionary,neg_filename_list,pos_filename_list,neg_path,pos_path,weight)
        print('When the feature type is :',feature,',the correct rate is:',correct_rate)
        print('')
        print('When the feature type is :',feature,',the top 10 positive features are:',positive_features)
        print('')
        print('When the feature type is :',feature,',the top 10 negative features are:',negative_features)
        print('')
