#==============================================================================
# Importing

import re
import operator
import pickle
import array
import time
import argparse
import os.path
import collections


#==============================================================================
# Command line processing

class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("infile", help="pass infile to huff-compress/decompress for compression/decompression")
        parser.add_argument("-s", "--symbolmodel", help="specify character- or word-based Huffman encoding -- default is character",
                            choices=["char","word"])
        args= parser.parse_args()

        self.file = args.infile
        self.symbolmodel=args.symbolmodel
        
        if not(self.symbolmodel):
            self.symbolmodel= "char"
            return 
        else:
            symbolmodel = self.symbolmodel
            return
#==============================================================================
# Get the probability of each word or char
def probability(file,RE):
    with open(file, 'r') as f:
        word_list=[]
        for line in f:
            term = re.findall(RE,line)
            for item in term:
                word_list.append(item) #Put every word in a list

    word_freq = dict(collections.Counter(word_list))#Use counter to get the frequency of every word

    sum_freq = sum(word_freq.values())#Get the sum of frequency
    
    #Get the probability of every word
    word_prob = {}
    for key,value in word_freq.items():
        word_prob[key]=value/sum_freq
        
    #Sort the probability of every word
    word_prob = dict(sorted(word_prob.items(),key=operator.itemgetter(1),reverse=True))

    #The reason why I return two result is next step will use word_list
    return word_prob,word_list
    
#==============================================================================
# Encoding of char
def char_encode(word_prob):
    #Put word and probability in different lists
    list_prob=[]
    list_word=[]
    for key,value in word_prob.items():
        list_word.append(key)
        list_prob.append(value)
   
    encode_dic={} #Result dictionary
    while 1:
        #Get the last two probs and words
        last1_prob = list_prob.pop()
        last1_word = list_word.pop()
        last2_prob = list_prob.pop()
        last2_word = list_word.pop()

        #Coding every word
        for i in last1_word:
            if i not in encode_dic:
                encode_dic[i]='0'
            else:
                encode_dic[i]='0'+encode_dic[i]
                
        for i in last2_word:
            if i not in encode_dic:
                encode_dic[i]='1'
            else:
                encode_dic[i]='1'+encode_dic[i]

        #Get the new prob and new word
        new_prob = last1_prob + last2_prob
        new_word = last1_word+last2_word
        
        #Sort the new prob list and word list 
        list_prob.append(new_prob)
        list_prob.sort()
        list_prob.reverse()
        index = list_prob.index(new_prob) #Find the new prob's index
        list_word.insert(index,new_word) #Put the new word in the new index
        
        #Break loop
        if len(list_prob)==1:
            break
    return encode_dic           

#==============================================================================
# Encoding of word
def word_encode(word_prob):
    #Put word and probability in different lists
    list_prob=[]
    list_word=[]
    for key,value in word_prob.items():
        list_word.append(key)
        list_prob.append(value)

    encode_dic={} #Result dictionary

    while 1:
        #Get the last two probs and words
        last1_prob = list_prob.pop()
        last1_word = list_word.pop()
        last2_prob = list_prob.pop()
        last2_word = list_word.pop()

        new_word = []
        #Coding every word
        if type(last1_word)==list:
            for i in last1_word:
                new_word.append(i) #Regroup a new word
                if i not in encode_dic:
                    encode_dic[i]='0'
                else:
                    encode_dic[i]='0'+encode_dic[i]
        else:
            new_word.append(last1_word)
            if last1_word not in encode_dic:
                encode_dic[last1_word]='0'
            else:
                encode_dic[last1_word]='0'+encode_dic[last1_word]
                
        if type(last2_word)==list:
            for i in last2_word:
                new_word.append(i)
                if i not in encode_dic:
                    encode_dic[i]='1'
                else:
                    encode_dic[i]='1'+encode_dic[i]
        else:
            new_word.append(last2_word)
            if last2_word not in encode_dic:
                encode_dic[last2_word]='1'
            else:
                encode_dic[last2_word]='1'+encode_dic[last2_word]

        #Get the new prob 
        new_prob = last1_prob + last2_prob
        
        #Order the new prob list and word list 
        list_prob.append(new_prob)
        list_prob.sort()
        list_prob.reverse()
        index = list_prob.index(new_prob) #Find the new prob's index
        list_word.insert(index,new_word) #Put the new word in the new index
        
        #Break loop
        if len(list_prob)==1:
            break

    return encode_dic

#==============================================================================
# Pickling the encoding
def pkl(root,encoding):
    pklname = root+'-symbol-model.pkl'
    with open(pklname, 'wb') as f:
        pickle.dump(encoding, f)

#==============================================================================
# Add the EOF
def EOF(word_list,encoding):
    encoding_str=''
    for i in word_list:
        encoding_str+=encoding[i]
    if len(encoding_str)%8!=0:
        remainder = len(encoding_str)%8 #Caculate the remainder of encoding_str since I should know how many code can not be exact division
        binary = bin(remainder).replace('0b','00') #Use binary code to represent the code which can not be exact division
        mark = '0'*(8-len(binary))+binary #Complete eight bits of binary

        EOF = encoding_str[-remainder:]+'0'*(8-remainder)#Use the encoding_str which are remainder of encoding_str to create complete binary
        encoding_str_EOF = encoding_str[0:len(encoding_str)-remainder]+mark+EOF#Concatenate the stop tag to the original encoding_str
        return encoding_str_EOF

    else:
        return encoding_str
    

#==============================================================================
#Create the bin file
def get_bin(encoding_str_EOF,root):
    codearray = array.array('B')
    c = ''
    for i in encoding_str_EOF:
        c=c+i
        if len(c)%8==0:
            b = int(c,2)
            codearray.append(b)
            c = ''
    binname = root+'.bin'
    f = open(binname, 'wb')
    codearray.tofile(f)
    f.close()

#==============================================================================
# MAIN

if __name__ == '__main__':
    
    config = CommandLine()

    
    #Get the name of input file without extension
    (root,file) = os.path.splitext(config.file)
    
    if config.symbolmodel == 'char':
        RE = r'\w|\W'
        start = time.clock()
        word_prob,word_list = probability(config.file,RE)
        end = time.clock()
        print('The time of building the symbol model : '+str(end-start)+' seconds')
        
        start = time.clock()
        encoding = char_encode(word_prob)
        end = time.clock()
        print('The time of encoding : '+str(end-start)+' seconds')

        pkl(root,encoding)
        encoding_str_EOF = EOF(word_list,encoding)
        get_bin(encoding_str_EOF,root)



    else:
        RE = r'[A-Za-z]+|\W|\d|_'
        start = time.clock()
        word_prob,word_list = probability(config.file,RE)
        end = time.clock()
        print('The time of building the symbol model : '+str(end-start)+' seconds')

        start = time.clock()
        encoding = word_encode(word_prob)
        end = time.clock()
        print('The time of encoding : '+str(end-start)+' seconds')
        
        pkl(root,encoding)
        encoding_str_EOF = EOF(word_list,encoding)
        get_bin(encoding_str_EOF,root)



