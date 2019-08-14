import os
from sklearn.metrics import accuracy_score,recall_score,f1_score

def evaluation(y_true,y_pred,result_dir):
    accuracy_score_result = accuracy_score(y_true, y_pred)
    recall_score_result = recall_score(y_true, y_pred,average='macro')
    f1_score_result = f1_score(y_true, y_pred,average = 'macro')
    print('The accuracy is:%f ' %accuracy_score_result)
    print('The macro_recall is:%f ' %recall_score_result)
    print('The macro_F_score is:%f ' %f1_score_result)
    with open(result_dir,"w") as f:
        f.write("The accuracy is : " + str(accuracy_score_result) )
        f.write('\n')
        f.write("The macro_recall is : " + str(recall_score_result) )
        f.write('\n')
        f.write("The macro_F_score is : " + str(f1_score_result) )
        f.write('\n')

def write_result(result_dir,grid_search):
    with open(result_dir,"w") as f:
        f.write("#################Training Record#################")
        f.write('\n')
        f.write(str(grid_search.cv_results_))
        f.write('\n')
        f.write("#################Best Score#################")
        f.write('\n')
        f.write(str(grid_search.best_score_))
        f.write('\n')
        f.write("#################Best Parameters#################")
        f.write('\n')
        f.write(str(grid_search.best_params_))
        f.write('\n')
        f.write("#################Best Classifier#################")
        f.write('\n')
        f.write(str(grid_search.best_estimator_))