#==============================================================================
# Importing
import argparse
import re
from collections import Counter

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
# Word Segmentation (Get the Unigrams and Bigrams)
def word_segmentation(trainfile,regex):
	Unigrams = []
	Bigrams = []
	Unigrams_count = 0
	Bigrams_count = 0 

	with open(trainfile) as f:
		for line in f.readlines():
			bag_of_words = ['<s>'] #Create a bag-of-words with first element '<s>' for next step 
			line = line.lower() #Change all words from uppercase to lowercase
			words = re.findall(regex,line) # Split sentecnes to words

			for word in words:
				bag_of_words.append(word) #Append every word to bag-of-words
			bag_of_words.append('<\s>') #Append last element '<\s>' at the end of bag-of-words

			for word in bag_of_words:
				Unigrams_count += 1 #Count the number of Unigram
				Unigrams.append(word) #Append every word to Unigrams to create Unigrams

			for i in range(len(bag_of_words) - 1):
				Bigrams_count += 1 #Count the number of Bigram
				bigram_word = bag_of_words[i] + ' ' + bag_of_words[i+1] #Create bigram_word 
				Bigrams.append(bigram_word)
	return Unigrams,Bigrams,Unigrams_count,Bigrams_count

#==============================================================================
# Get the probability of every word in Unigrams and Bigrams
def get_probability(Unigrams,Bigrams,Unigrams_count,Bigrams_count):
	Unigrams_pro = {} #Create a dictionary for probability of Unigrams
	Bigrams_pro = {} #Create a dictionary for probability of Bigrams
	Bigrams_pro_smooth = {} #Create a dictionary for probability of Add-1 smooth Bigrams
	Unigrams_fre = dict(Counter(Unigrams)) #Get the frequence of every word in Unigrams
	Bigrams_fre = dict(Counter(Bigrams)) #Get the frequence of every word in Bigrams
	for key,value in Unigrams_fre.items():
		Unigrams_pro[key] = value/Unigrams_count #Get the probability of every word in Unigrams

	for key,value in Bigrams_fre.items():
		previous_word = key.split()[0] #Because the space between two words, I can use 'split' to get the first word(i.e. previous word)
		Bigrams_pro[key] = value/Unigrams_fre[previous_word]

	for key,value in Bigrams_fre.items():
		previous_word = key.split()[0] #Because the space between two words, I can use 'split' to get the first word(i.e. previous word)
		Bigrams_pro_smooth[key] = (value + 1 ) / (Unigrams_fre[previous_word] + Unigrams_count) #Add-1 smoothing 

	return Unigrams_pro,Unigrams_fre,Bigrams_pro,Bigrams_pro_smooth

#==============================================================================
# Get the structured sentence from question.text (List nesting format)
def get_structured_sentence(testfile,regex):
	with open(testfile) as f:
		structured_sentence_list = [] #Create a list to store the candidate structured sentences
		for line in f.readlines():
			candidate_sentence_list = [] #Create a list to store two candidate structured sentences
			line = line.lower()
			bag_of_words = re.findall(regex,line) 

			first_candidate_word = bag_of_words.pop(-1) #Delete the first candidate word 
			second_candidate_word = bag_of_words.pop(-1) #Delete the second candidate word

			first_sentence_list = list(bag_of_words) #Create a list of first candidate sentence
			second_sentence_list = list(bag_of_words) #Create a list of second candidate sentence

			index_of____ = bag_of_words.index('____') #Get the index of '____' in bag_of_words
			first_sentence_list[index_of____] = first_candidate_word #Replace the '____' with the first candidate word in first sentence 
			second_sentence_list[index_of____] = second_candidate_word #Replace the '____' with the second candidate word in second sentence 

			first_sentence_list.insert(0,'<s>') #Insert '<s>' into the first index of first_sentence_list
			first_sentence_list.append('<\s>') #Append '<\s>' into first_sentence_list
			second_sentence_list.insert(0,'<s>') #Insert '<s>' into the first index of second_sentence_list
			second_sentence_list.append('<\s>') #Append '<\s>' into second_sentence_list

			candidate_sentence_list.append(first_sentence_list)
			candidate_sentence_list.append(second_sentence_list)
			structured_sentence_list.append(candidate_sentence_list) #List nesting format

	return structured_sentence_list

#==============================================================================
# Get the answer by Unigrams model
def unigrams_model(structured_sentence_list,Unigrams_pro):
	answer = [] #Create a list to store unigram-model's answer
	answer_sentence = []
	for candidate_sentence in structured_sentence_list:
		score_list = [] #Create a list to store the scores of every candidate 
		for candidate in candidate_sentence:
			score = 1 #Initialize the score with 1
			for unigram in candidate:
				score = score * Unigrams_pro[unigram] #Compute the score of this candidate
			score_list.append(score) #Append this score to score list
		if score_list[0] == score_list[1] and score_list[0] == 0 : #If the value of two scores are same and equal to 0, append 2 to represent this.
			answer.append(2)
		elif score_list[0] == score_list[1] and score_list[0] != 0: #If the value of two scores are same and don't equal to 0, append 3 to represent this.
			answer.append(3)
		else:
			answer.append(score_list.index(max(score_list))) #Append the index of biggest score of candidates from score_list to answer(Using index to repersent the answer)	
	return answer

#==============================================================================
# Get the answer by Bigrams model
def bigrams_model(structured_sentence_list,Bigrams_pro):
	answer = [] #Create a list to store unigram-model's answer
	for candidate_sentence in structured_sentence_list:
		score_list = [] #Create a list to store the scores of every candidate 
		for candidate in candidate_sentence:
			score = 1 #Initialize the score with 1
			for i in range(len(candidate) - 1):
				bigram = candidate[i] + ' ' + candidate[i+1] #Create bigram word
				if bigram in Bigrams_pro:
					score = score*Bigrams_pro[bigram] #If bigram word in Bigrams_pro, use the probability of Bigrams_pro 
				else:
					score = 0 #If not, set the score to 0
			score_list.append(score) #Append this score to score list
		if score_list[0] == score_list[1] and score_list[0] == 0 : #If the value of two scores are same and equal to 0, append 2 to represent this.
			answer.append(2)
		elif score_list[0] == score_list[1] and score_list[0] != 0: #If the value of two scores are same and don't equal to 0, append 3 to represent this.
			answer.append(3)
		else:
			answer.append(score_list.index(max(score_list))) #Append the index of biggest score of candidates from score_list to answer(Using index to repersent the answer)
	return answer

#==============================================================================
# Get the answer by Bigrams with smooth model
def bigrams_smooth_model(structured_sentence_list,Bigrams_pro_smooth,Unigrams_fre,Unigrams_count):
	answer = [] #Create a list to store unigram-model's answer
	for candidate_sentence in structured_sentence_list:
		score_list = [] #Create a list to store the scores of every candidate 
		for candidate in candidate_sentence:
			score = 1 #Initialize the score with 1
			for i in range(len(candidate) - 1):
				bigram = candidate[i] + ' ' + candidate[i+1] #Create bigram word
				if bigram in Bigrams_pro:
					score = score*Bigrams_pro_smooth[bigram] #If bigram word in Bigrams_pro_smooth, directly use the probability of Bigrams_pro_smooth 
				else:
					previous_word = bigram.split()[0] #If not, because the space between two words, I can use 'split' to get the first word(i.e. previous word)
					Bigrams_pro_smooth[bigram] = 1/(Unigrams_fre[previous_word] + Unigrams_count) #Add-1 smooth to new bigram
					score = score*Bigrams_pro_smooth[bigram] 
			score_list.append(score) #Append this score to score list
		if score_list[0] == score_list[1] and score_list[0] == 0 : #If the value of two scores are same and equal to 0, append 2 to represent this.
			answer.append(2)
		elif score_list[0] == score_list[1] and score_list[0] != 0: #If the value of two scores are same and don't equal to 0, append 3 to represent this.
			answer.append(3)
		else:
			answer.append(score_list.index(max(score_list))) #Append the index of biggest score of candidates from score_list to answer(Using index to repersent the answer)
	return answer

#==============================================================================
# Compute the correct rate
def compute_correct_rate(answer,strandard_answer):
	correct_count = 0 #Initialize the number of correct candidate with 1
	for i in range(len(answer)):
		if answer[i] == strandard_answer[i]:
			correct_count += 1 #Plus 1 if the answer is correct (i.e. same as given strandard answer)
		elif answer[i] == 3:
			correct_count += 0.5 #Plus 0.5 because it returns two equal non-zero probabilities
	correct_rate = correct_count/len(answer) #Compute the correct rate by using the number of correct answer divide the amount
	return correct_rate

#==============================================================================
# MAIN
if __name__ == '__main__':
    config = CommandLine()
    regex = r"[a-zA-Z]+-[a-zA-Z]+|[a-zA-Z]+|'[a-zA-Z]+"  #Regex for training text
    regex_question = r"[a-zA-Z]+-[a-zA-Z]+|[a-zA-Z]+|'[a-zA-Z]+|____" #Regex for question text

    Unigrams,Bigrams,Unigrams_count,Bigrams_count = word_segmentation(config.trainfile,regex) #Call the function word_segmentation to do word Segmentation
    Unigrams_pro,Unigrams_fre,Bigrams_pro,Bigrams_pro_smooth = get_probability(Unigrams,Bigrams,Unigrams_count,Bigrams_count) #Call the function get_probability to get the probabilities of Unigrams,Bigrams and Bigrams with smoothing
    structured_sentence_list = get_structured_sentence(config.testfile,regex_question) #Call the function get_structured_sentence to get the structured candidate sentences

    unigrams_answer = unigrams_model(structured_sentence_list,Unigrams_pro) #Call the function unigram_model to get the answer by unigram model
    bigrams_answer = bigrams_model(structured_sentence_list,Bigrams_pro) #Call the function bigram_model to get the answer by bigram model
    bigrams_smooth_answer = bigrams_smooth_model(structured_sentence_list,Bigrams_pro_smooth,Unigrams_fre,Unigrams_count) #Call the function bigram_smooth_model to get the answer by bigram with smoothing model 

    strandard_answer = [0,1,1,0,0,0,1,0,1,0] #Manually create the strandard answer 

    unigram_correct_rate = compute_correct_rate(unigrams_answer,strandard_answer) #Call the function compute_correct_rate to get the accuracy of unigram language model
    print('The accuracy of unigram language model is :',unigram_correct_rate)
    bigram_correct_rate = compute_correct_rate(bigrams_answer,strandard_answer) #Call the function compute_correct_rate to get the accuracy of bigram language model
    print('The accuracy of bigram language model is :',bigram_correct_rate)
    bigram_smooth_correct_rate = compute_correct_rate(bigrams_smooth_answer,strandard_answer) #Call the function compute_correct_rate to get the accuracy of bigram with add-1 smoothing language model
    print('The accuracy of bigram with add-1 smoothing language model is :',bigram_smooth_correct_rate)