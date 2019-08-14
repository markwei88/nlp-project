import numpy as np
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def load_word_embedding(path):
    word_embedding = {}
    with open(path,encoding='utf8') as word_vector:
        for line in word_vector:
            value = line.split()
            word = value[0]
            vector = [float(i) for i in value[1:]]
            length = len(vector)
            vector = np.asarray(vector)
            word_embedding[word] = vector.reshape(1,length)
    return word_embedding

def padding(X,average_len):
    if len(X) >= average_len:
        X = X[:average_len]
    else:
        X = X + ['<pad>']*(average_len - len(X))
    return X

def get_vacabulary(x):
    vocab = []
    for sent in x:
        vocab += sent
    vocab = list(set(vocab))
    vocab = ['<unknown>'] + vocab
    vocab = ['<pad>'] + vocab
    return vocab

def pre_process(x_train,x_val,x_test):
    #Do tokenization,word segmentation,spell correct,annotate
    X_train = [text_processor.pre_process_doc(sent) for sent in x_train]
    X_val = [text_processor.pre_process_doc(sent) for sent in x_val]
    X_test = [text_processor.pre_process_doc(sent) for sent in x_test]
    #Get the average length of all data
    length_list = [len(sent) for sent in X_train] + [len(sent) for sent in X_val] + [len(sent) for sent in X_test]
    average_len = sum(length_list)//(len(X_train) + len(X_val) + len(X_test))
    #Get the vocab of training set
    vocab = get_vacabulary(X_train)
    #Unified length
    X_train = [padding(sent,average_len) for sent in X_train]
    X_val = [padding(sent,average_len) for sent in X_val]
    X_test = [padding(sent,average_len) for sent in X_test]

    return X_train,X_val,X_test,average_len,vocab