import os
import csv
import argparse
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from data_loader import get_data
from pre_processing import Preprocess
from record_result import write_result,evaluation
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model", default='LR', type=str,
                        help="Model selection"
                        "Options: LR,SVM,NB,DT,RF"
                        "Logistic Regression, Support Vector Machine, Naive Bayes, Decision Tree, Random Forest")
    parser.add_argument("--features", default=None, type=str, required=True,
                        help="Features selection"
                        "Options: ct, tfidf, wb"
                        "Counts, TF-IDF, Word-Embedding")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and the best model parameters will be written.")

    ## Optional parameters
    parser.add_argument("--n_gram",
                        default=1,
                        type=int,
                        help="The choose of n-gram, 1 means unigram, 2 means unigram+bigram, 3 means unigram+bigram+trigram...")
    parser.add_argument("--remove_punctuation",
                        default=True,
                        type=bool,
                        help="Whether to remove punctuations")
    parser.add_argument("--remove_stop_list",
                        default=False,
                        type=bool,
                        help="Whether to remove stop words list ")
    parser.add_argument("--do_stemming",
                        default=True,
                        type=bool,
                        help="Whether to do stemming")
    parser.add_argument("--do_lemmatization",
                        default=True,
                        type=bool,
                        help="Whether to do lemmatization")
    parser.add_argument("--do_word_segmentation",
                        default=False,
                        type=bool,
                        help="Whether to do word segmentation")
    parser.add_argument("--deal_with_others",
                        default=True,
                        type=bool,
                        help="Whether to deal with numbers/url/some words start with")
    parser.add_argument("--preprocess_method",
                        default='self_preprocess',
                        type=str,
                        help="The choose of preprocess method"
                        "self_preprocess,datastories_preprocess")

    args = parser.parse_args()

################### Loading data ###################
    x_trainval,y_trainval,x_test,y_test = get_data(args.data_dir)

################### Preprocess data ###################
    if args.preprocess_method == 'self_preprocess':

        pre_process_object_train = Preprocess(x_trainval)
        pre_process_object_test = Preprocess(x_test)

        if args.deal_with_others:
            pre_process_object_train.deal_with_others()
            pre_process_object_test.deal_with_others()

        if args.remove_punctuation:
            pre_process_object_train.remove_punctuation()
            pre_process_object_test.remove_punctuation()

        if args.do_stemming:
            pre_process_object_train.stemming()
            pre_process_object_test.stemming()

        if args.do_lemmatization:
            pre_process_object_train.lemmatizer()
            pre_process_object_test.lemmatizer()

        # if args.do_word_segmentation:
        #     pre_process_object.segmentation()

        x_trainval = pre_process_object_train.text
        x_test = pre_process_object_test.text

    elif args.preprocess_method == 'datastories_preprocess':
        from pre_processing import datastories_processor
        x_trainval = datastories_processor(x_trainval)
        x_test = datastories_processor(x_test)

 ################### Create GridSearch to find the best model ###################   
    if args.features == 'ct':
        FeatureVectorizer = CountVectorizer()

    elif args.features == 'tfidf':
        FeatureVectorizer = TfidfVectorizer()
          
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=66666)

    gram_list = [(1,1),(1,2),(1,3),(1,4),(1,5)]
    stop_list = [None,'english']
    max_df = [0.8,1.0]
    min_df = [1,3,5]

    pipeline_obj = Pipeline([('vect',FeatureVectorizer),('clf',LogisticRegression())])

    with open('LR_pre_process_result_66666.csv', 'w') as csvfile:
        fieldnames = ['N_gram', 'Stop_List', 'Max_df','Min_df','Accuracy', 'Macro_Recall', 'Macro_F1','Accuracy_Test','Macro_Recall_Test','Macro_F1_Test']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for gram in gram_list:
            for stop in stop_list:
                for x_df in max_df:
                    for i_df in min_df:
                        pipeline_obj.set_params(clf__solver = 'sag',clf__class_weight = 'balanced',vect__ngram_range = gram,vect__stop_words = stop,vect__max_df=x_df,vect__min_df=i_df)
                        pipeline_obj.fit(x_train,y_train)
                        y_pred= pipeline_obj.predict(x_val)
                        accuracy_score_result = accuracy_score(y_val, y_pred)
                        recall_score_result = recall_score(y_val, y_pred,average='macro')
                        f1_score_result = f1_score(y_val, y_pred,average = 'macro')

                        y_pred_test= pipeline_obj.predict(x_test)
                        accuracy_score_result_test = accuracy_score(y_test, y_pred_test)
                        recall_score_result_test = recall_score(y_test, y_pred_test,average='macro')
                        f1_score_result_test = f1_score(y_test, y_pred_test,average = 'macro')

                        writer.writerow({'N_gram': gram, 
                                        'Stop_List': stop, 
                                        'Max_df' : x_df,
                                        'Min_df': i_df,
                                        'Accuracy':accuracy_score_result, 
                                        'Macro_Recall':recall_score_result, 
                                        'Macro_F1':f1_score_result,
                                        'Accuracy_Test':accuracy_score_result_test, 
                                        'Macro_Recall_Test':recall_score_result_test, 
                                        'Macro_F1_Test':f1_score_result_test})

if __name__ == "__main__":
    main()