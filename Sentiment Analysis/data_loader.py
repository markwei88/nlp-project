import os
from sklearn.model_selection import train_test_split

def get_data_file_path(path):
    for path,foldername,filename in os.walk(path):
        return [os.path.join(path,file_name) for file_name in filename]

def load_data(path):
    with open(path,encoding="utf8") as file:
        data = file.read()\
            .replace('neutral','0')\
            .replace('positive','1')\
            .replace('negative','2')\
            .split('\n')
        if data[-1] == '':
            data.pop()
        label = [int(sent.split('\t')[1]) for sent in data]
        text = [sent.split('\t')[2] for sent in data]

    return text,label    

def merge_data(data_file_path):
    text = []
    label = []
    for path in data_file_path:
        x,y = load_data(path)
        text += x
        label += y

    return text,label

def get_data(data_dir):
    train_path = os.path.join(data_dir,'train')
    test_path = os.path.join(data_dir,'test\SemEval2017-task4-test.subtask-A.english.txt')
    data_file_path = get_data_file_path(train_path)
    #Load all training data from different file(including training set and validation set)
    x_trainval, y_trainval = merge_data(data_file_path)
    #Load test data
    x_test,y_test = load_data(test_path)

    return x_trainval,y_trainval,x_test,y_test  