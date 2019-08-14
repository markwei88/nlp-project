import numpy as np
import operator
import re
import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.dic_document = {}
        self.dic_tfidf = {}

        #Create a dictionary and put items into documents where they are in
        for key,value in self.index.items():
            for key_v,value_v in value.items():
                if key_v not in self.dic_document:
                    self.dic_document[key_v]={}
                if key not in self.dic_document[key_v]:
                    self.dic_document[key_v][key]=value_v

        self.d = len(self.dic_document) #The total number of documents in the collection
        
        #(Only use for when method is TFIDF)Caculate the value of TFIDF of every items of documents
        for i in self.dic_document:  
            self.dic_tfidf[i]={}
            for key,value in self.dic_document[i].items():
                tf = value
                df = len(self.index[key])
                idf = math.log10(self.d/df)
                tfidf = tf*idf
                self.dic_tfidf[i][key]=tfidf


    # Method performing retrieval for specified query
    
    #Define candidate function to put candidate documents in a dictionary
    def candidate(self,query):
        dic_candidate = {}
        for q in query:
            if q in self.index:
                for key,value in self.index[q].items():
                    if key not in dic_candidate:
                        dic_candidate[key] = {}
                    if q not in dic_candidate[key]:
                        dic_candidate[key][q]=value
        return(dic_candidate)
    
    #The method of binary
    def binary(self,query):
        dic_candidate = self.candidate(query)
        result_dic = {}
        for key,value in dic_candidate.items():
            d2 = (len(self.dic_document[key]))**0.5
            qd = len(value)
            sim = qd/d2
            result_dic[key]=sim
        sorted_result_dic = sorted(result_dic.items(),key=operator.itemgetter(1),reverse=True)
        
        list_result = []            
        for r in range(len(sorted_result_dic)):
            list_result.append(sorted_result_dic[r][0])
        return(list_result)
    
    #The method of term frequency
    def tf(self,query):
        dic_candidate = self.candidate(query)
        result_dic = {}
        for key,value in dic_candidate.items():
            sum_qd = 0
            for k,v in value.items():
                q = query[k]
                qd = q*v
                sum_qd = sum_qd + qd
                
            sum_d2 = 0 
            for k,v in self.dic_document[key].items():
                d2 = v**2
                sum_d2 = sum_d2 + d2
            sqrt_sum = sum_d2**0.5 #The size of candidate document vector
            
            result = sum_qd/sqrt_sum    
            result_dic[key]=result
        sorted_result_dic = sorted(result_dic.items(),key=operator.itemgetter(1),reverse=True)

        list_result = []            
        for r in range(len(sorted_result_dic)):
            list_result.append(sorted_result_dic[r][0])
        return(list_result)

    #The method of TFIDF
    def tfidf(self,query):
        #Calculate the value of TFIDF of every items in query and put them in dictionary
        dic_q_tfidf = {}
        for key,value in query.items():
            if key in self.index:
                df = len(self.index[key])
                idf = math.log10(self.d/df)
                tfidf = value*idf
                dic_q_tfidf[key]=tfidf

        #Calculate the value of TFIDF of every items in candidate document and put them in dictionary
        dic_d_tfidf = {}
        for q in query:
            if q in self.index:
                for k,v in self.index[q].items():
                    if k not in dic_d_tfidf :
                        dic_d_tfidf[k] = {}
                    if q not in dic_d_tfidf [k]:
                        dic_d_tfidf[k][q]=self.dic_tfidf[k][q]

        dic_result = {}
        for key,value in dic_d_tfidf.items():
            list_d2 = []
            for k,v in self.dic_tfidf[key].items():
                d2 = v**2
                list_d2.append(d2)
            sqrt_d2 = sum(list_d2)**0.5 #The size of candidate document vector

            sum_qd = 0
            for k,v in value.items():
                qd = dic_q_tfidf[k]*v
                sum_qd = sum_qd + qd 
            
            sim = sum_qd/sqrt_d2
            dic_result[key] = sim
            
        sorted_dic_result = sorted(dic_result.items(),key=operator.itemgetter(1),reverse=True)
        list_result = []            
        for r in range(len(sorted_dic_result)):
            list_result.append(sorted_dic_result[r][0])
        return(list_result)

    def forQuery(self, query):
        #Call the functions of binary, tf or tfidf
        if self.termWeighting == 'binary':
            result = self.binary(query)
            return(result)

        elif self.termWeighting == 'tf':
            result = self.tf(query)
            return(result)

        else:
            result = self.tfidf(query)
            return(result)
