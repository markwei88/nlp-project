import numpy as np
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