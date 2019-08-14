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
# #From binary convert to str
def get_str(binary):
    encoding_str = ''
    for i in binary:
        i = str(i)
        c=bin(int(i, 10))
        c=c[2:]
        if len(c)==8:
            encoding_str+=c
        else:
            c = (8-len(c))*'0'+c
            encoding_str+=c
    return encoding_str

#==============================================================================
# Process EOF
def del_EOF(encoding_str):
    mark = encoding_str[-16:]
    encoding_str = encoding_str[:-16]
    stop_tag = mark[:8]
    remainder = int(stop_tag,2)
    orginial = mark[8:][:remainder]
    orginial_code = encoding_str+orginial
    return orginial_code


#==============================================================================
# Decode
def decode(orginial,encoding):
    decode_result = ''
    start = 0
    for i in range(len(orginial)+1):
        if orginial[start:i] in encoding:
            decode_result+=encoding[orginial[start:i]]
            start = i
    return decode_result

#==============================================================================
# Create the file of decode
def create_file(decode_result,root):
    f = open(root+'-decompressed.txt', 'w')
    f.write(decode_result)
    f.close()

#==============================================================================
# MAIN

if __name__ == '__main__':
    
    config = CommandLine()

    
    #Get the name of input file without extension
    (root,file) = os.path.splitext(config.file)
    
    if config.symbolmodel == 'char':
        start = time.clock()
        with open(root+'-symbol-model.pkl', 'rb') as f:
            encoding = pickle.load(f)
        with open(root+'.bin', 'rb') as f:
            binary = f.read()

        encoding_str = get_str(binary)
        orginial_code = del_EOF(encoding_str)
        
        #Interchange of values and keys of encoding
        new_encoding = {}
        for key,value in encoding.items():
            new_encoding[value]=key

        decode_result = decode(orginial_code,new_encoding)
        end = time.clock()
        print('The time of decoding the compressed file : '+str(end-start)+' seconds')
        
        create_file(decode_result,root)
            
    else:
        start = time.clock()
        with open(root+'-symbol-model.pkl', 'rb') as f:
            encoding = pickle.load(f)
        with open(root+'.bin', 'rb') as f:
            binary = f.read()

        encoding_str = get_str(binary)
        orginial_code = del_EOF(encoding_str)
        
        #Interchange of values and keys of encoding
        new_encoding = {}
        for key,value in encoding.items():
            new_encoding[value]=key

        decode_result = decode(orginial_code,new_encoding)
        end = time.clock()
        print('The time of decoding the compressed file : '+str(end-start)+' seconds')
        
        create_file(decode_result,root)
            


