# Data manipulation
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile

# Parsing and pre-processing
import glob, re, os, sys, random
from random import shuffle
import random
from time import time

from langdetect import detect, DetectorFactory

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# Vector representations and embeddings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

# Modeling - Logistic, XGBOOST, SVM
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, roc_auc_score, auc
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from sklearn.pipeline import Pipeline, FeatureUnion

from xgboost import XGBClassifier
import pickle


########################################################################################################
# function to create label column
def create_label(data, label_name):
    df = data.copy()

    df1 = df[df[label_name].isin([0,1])].reset_index(drop=True)

    le = LabelEncoder()
    df1['label'] = le.fit_transform(df1[label_name])

    return(df1)

########################################################################################################
# define gridsearch cv
def gridsearch(pipeline, parameters, X_train, y_train, cv, scoring):
    # Create a GridSearchCV object with the pipeline and hyperparameters
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=24, verbose=1, scoring=scoring)
    t0 = time()

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    print("done in %0.3fs" % (time() - t0))

    # Print the best hyperparameters and the corresponding mean cross-validated score
    print("Best cross-validation score: ", grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return(grid_search, best_parameters)

########################################################################################################
# function to evaluate performance
def evaluate(Ytest, Ypredict): #evaluate the model (accuracy, precision, recall, f-score, confusion matrix)
    print('Accuracy:', accuracy_score(Ytest, Ypredict) )
    print('\nClassification report:\n', classification_report(Ytest, Ypredict))
    print('\nCR:', precision_recall_fscore_support(Ytest, Ypredict, average='macro'))
    print('\nConfusion matrix:\n', confusion_matrix(Ytest, Ypredict), '\n\n_______________________\n\n')
     # [[tn fp]
     # [fn tp]]

    # Evaluate the performance of the model
    tn, fp, fn, tp = confusion_matrix(Ytest, Ypredict).ravel()
    fpr = fp / (fp + tn)
    accuracy = accuracy_score(Ytest, Ypredict) *100.0
    precision = precision_score(Ytest, Ypredict, average='binary')
    recall = recall_score(Ytest, Ypredict, average='binary')
    f_score = 2 * (precision * recall) / (precision + recall)
    roc_auc = roc_auc_score(Ytest, Ypredict, average=None)

    print(f'Recall: {recall:.3f} \n Precision: {precision:.3f} \n F1: {f_score:.3f} \n FPR: {fpr:.3f} \n Accuracy: {accuracy:.3f} \n ROC_AUC: {roc_auc:.3f}')
    
    return([tn, fp, fn, tp, recall, precision, f_score, fpr, accuracy, roc_auc])

# fit logit using best params in train set
# c = 5
# solver = 'liblinear'
# model = LogisticRegression(C=c, solver = solver)
def fit_best_model_train(X_train, y_train, model, best_parameters, cv): #cv =3

    vec = TfidfVectorizer()
    clf = model
    pipeline_test = Pipeline([('tfidf', vec), ('clf', clf)])
    pipeline_test.set_params(**best_parameters)
    print('fitting the best model')
    pipeline_test.fit(X_train, y_train)

    y_predict_train = cross_val_predict(pipeline_test, X_train, y_train, cv=cv)
    evaluate(y_train, y_predict_train)

    return(pipeline_test, y_predict_train)

def fit_best_model_test(X_test, y_test, pipeline_test):
    
    print('testing on test set')
    y_predict_test = pipeline_test.predict(X_test)
    evaluate(y_test, y_predict_test)
    
    return(y_predict_test)
    
#######################################################################################################
# create df of feature importances
def get_feature_importance_cv(pipeline_name):
    # Assuming your model is called "pipeline_cv" and you have access to the feature importances
    feature_importances = pipeline_name.named_steps['clf'].coef_[0]
    feature_names = pipeline_name.named_steps['tfidf'].get_feature_names_out()

    # Create a dataframe with feature names and their importances
    df_features = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    df_features = df_features.reindex(df_features['importance'].abs().sort_values(ascending=False).index)

    return(df_features)


########################################################################################################
# evaluate confusion matrix
def eval_matrix(df, X, y_true, y_pred):
    results = pd.DataFrame(np.column_stack((X, y_true, y_pred)), columns=['text', 'target', 'y_predict'])
    results = pd.concat([df, results], axis=1)

    tn=results[(results.target == 0) & (results.y_predict == 0)]
    fp=results[(results.target == 0) & (results.y_predict == 1)]
    fn=results[(results.target == 1) & (results.y_predict == 0)]
    tp=results[(results.target == 1) & (results.y_predict == 1)]

    return (results, tn, fp, fn, tp)
















