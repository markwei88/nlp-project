import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score,confusion_matrix

def categoryFromOutput(output,id_to_target):
    top_n, top_i = output.topk(1)
    result = [id_to_target[i] for i in top_i[:,0].tolist()]   
    return result

def evaluate(loader,model,id_to_target):
    y_pred = []
    y_true = []
    for x,y in loader:
        if torch.cuda.is_available():
            x,y = x.cuda(),y.cuda()
        output = model(x)
        result = categoryFromOutput(output,id_to_target)
        y_pred += result
        result_real = [id_to_target[i] for i in y[:,0].tolist()] 
        y_true += result_real
    # print('The accuracy is:%f ' %accuracy_score(y_true, y_pred))
    # print('The macro_recall is:%f ' %recall_score(y_true, y_pred,average='macro'))
    # print('The macro_F_score is:%f ' %f1_score(y_true, y_pred,average = 'macro'))
    return accuracy_score(y_true, y_pred),recall_score(y_true, y_pred,average='macro'),f1_score(y_true, y_pred,average = 'macro')