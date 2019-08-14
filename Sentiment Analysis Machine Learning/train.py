import os
import argparse
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.svm import SVC
from data_loader import get_data
from pre_processing import Preprocess
from record_result import write_result,evaluation
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
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
    parser.add_argument("--model", default=None, type=str, required=True,
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
   
    if args.model == 'LR':           
        pipeline_obj = Pipeline([('feature',FeatureVectorizer),('clf',LogisticRegression())])
        parameters = {'feature__stop_words': (None,'english'),'feature__ngram_range': ((1,1),(1,2),(1,3),(1,4),(1,5)),'feature__max_df':(0.8,1.0),'feature__min_df':(1,3,5),
                    'clf__solver': ['sag'],'clf__class_weight':['balanced']}
        grid_search = GridSearchCV(pipeline_obj, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_trainval,y_trainval)

    elif args.model == 'SVM':           
        pipeline_obj = Pipeline([('feature',FeatureVectorizer),('clf',svm.LinearSVC())])
        parameters = {'feature__stop_words': [None],'feature__ngram_range': [(1,2)],'feature__max_df':[0.8],'feature__min_df':[1],
                    'clf__C': [0.01],'clf__class_weight':['balanced']}
        grid_search = GridSearchCV(pipeline_obj, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_trainval,y_trainval)

    elif args.model == 'SVC':           
        pipeline_obj = Pipeline([('feature',FeatureVectorizer),('clf',svm.SVC())])
        parameters = {'feature__stop_words': [None],'feature__ngram_range': [(1,2)],'feature__max_df':[0.8],'feature__min_df':[1],
                    'clf__C': [0.0001,0.0005,0.001,0.005,0.01],'clf__class_weight':['balanced'],'clf__decision_function_shape':['ovo'],}
        grid_search = GridSearchCV(pipeline_obj, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_trainval,y_trainval)

    elif args.model == 'NB':           
        pipeline_obj = Pipeline([('feature',FeatureVectorizer),('clf',MultinomialNB())])
        parameters = {'feature__stop_words': [None],'feature__ngram_range': [(1,2)],'feature__max_df':[0.8],'feature__min_df':[1],
                    'clf__alpha': [0.1]}
        grid_search = GridSearchCV(pipeline_obj, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_trainval,y_trainval)

    elif args.model == 'DT':           
        pipeline_obj = Pipeline([('feature',FeatureVectorizer),('clf',DecisionTreeClassifier())])
        parameters = {'feature__stop_words': [None],'feature__ngram_range': [(1,2)],'feature__max_df':[0.8],'feature__min_df':[1],
                    'clf__criterion': ['gini'],'clf__class_weight':['balanced'],'clf__max_depth':(100,500)}
        grid_search = GridSearchCV(pipeline_obj, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_trainval,y_trainval)

    elif args.model == 'RF':           
        pipeline_obj = Pipeline([('feature',FeatureVectorizer),('clf',DecisionTreeClassifier())])
        parameters = {'feature__stop_words': [None],'feature__ngram_range': [(1,2)],'feature__max_df':[0.8],'feature__min_df':[1],
                    'clf__class_weight':['balanced'],'clf__max_depth':(100,300,500),'clf__min_samples_split':(100,300,500)}
        grid_search = GridSearchCV(pipeline_obj, parameters, cv=3, scoring='f1_macro')
        grid_search.fit(x_trainval,y_trainval)


    #Do predition and evaluation
    best_model=grid_search.best_estimator_
    y_pred=best_model.predict(x_test)
    evaluation(y_test,y_pred,os.path.join(args.output_dir, args.model + 'test_result.txt'))

    print(grid_search)
    print(' ')
    print(grid_search.best_score_)
    print(' ')
    print(grid_search.best_params_)

    
    write_result(os.path.join(args.output_dir, args.model + 'train_result.txt'), grid_search)


if __name__ == "__main__":
    main()