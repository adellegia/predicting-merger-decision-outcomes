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

    print(f' Accuracy: {accuracy:.3f} \n Precision: {precision:.3f} \n Recall: {recall:.3f} \n F1: {f_score:.3f} \n FPR: {fpr:.3f} \n ROC_AUC: {roc_auc:.3f}')

########################################################################################################
# train
# c = 5
# solver = 'liblinear'
# model = LogisticRegression(C=c, solver = solver)

def train_model_cross_val(Xtrain, Ytrain, vec, model, cv):
    print('***10-fold cross-validation***')
    pipeline = Pipeline([
        ('features', FeatureUnion([vec])),
        ('classifier', model)
    ])
    Ypredict_train = cross_val_predict(pipeline, Xtrain, Ytrain, cv=cv)

    pipeline.fit(Xtrain, Ytrain)  # fit the pipeline on the whole training data for feature importance
    trained_model = pipeline.named_steps['classifier']

    evaluate(Ytrain, Ypredict_train)
    return pipeline, trained_model, Ypredict_train


########################################################################################################
# test
def train_model_test(Xtrain, Ytrain, Xtest_v, Ytest_v, model, vec): #test on 'violations' test set
    pipeline = Pipeline([
        ('features', FeatureUnion([vec])),
        ('classifier', model)])
    pipeline.fit(Xtrain, Ytrain)
    print('***testing on test set***')

    Ypredict_test = pipeline.predict(Xtest_v) # fit the pipeline on the whole test set
    #tested_model = pipeline.named_steps['classifier']

    evaluate(Ytest_v, Ypredict_test)
    return(pipeline, Ypredict_test)

#######################################################################################################
# create df of feature importances
def get_feature_importance(trained_model, pipeline_train):
    # Get the feature importance values from the trained model
    feature_importance = trained_model.coef_[0]

    # Map the feature importance values to the feature names
    feature_names = pipeline_train.named_steps['features'].transformer_list[0][1].get_feature_names_out()
    feature_importance_map = dict(zip(feature_names, feature_importance))

    # Create a DataFrame from the feature importance map
    df_feature_importance = pd.DataFrame.from_dict(feature_importance_map, orient='index', columns=['importance'])
    df_feature_importance = df_feature_importance.reset_index()

    df_feature_importance = df_feature_importance.rename({'index':'feature'}, axis=1)

    # Sort the DataFrame by the absolute value of the importance score
    df_feature_importance['abs_importance'] = abs(df_feature_importance['importance'])
    df_feature_importance = df_feature_importance.sort_values(by='abs_importance', ascending=False)

    # Print the feature importance values
    # print(df_feature_importance)
    
    return (df_feature_importance, feature_importance, feature_names)


#######################################################################################################
# create df of feature importances by label
def get_feature_importance10(df_features_train):

    # label=1
    df_feature_importance1 = df_features_train[df_features_train['importance']>0].sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # label=0
    df_feature_importance0 = df_features_train[df_features_train['importance']<0].reset_index(drop=True)
    df_feature_importance0['abs_importance'] = abs(df_feature_importance0['importance'])
    df_feature_importance0 = df_feature_importance0.sort_values(by='abs_importance', ascending=False)

    return(df_feature_importance1, df_feature_importance0)

#######################################################################################################
# evaluate confusion matrix
def eval_matrix(df, merger_info, X, y_true, y_pred):
    results = pd.DataFrame(np.column_stack((X, y_true, y_pred)), columns=['text', 'target', 'y_predict'])
    results = pd.concat([df, results], axis=1)
    results = pd.merge(results, merger_info, how='left', left_on=['case_num', 'article'], right_on=['case_code', 'article'])

    tn=results[(results.target == 0) & (results.y_predict == 0)]
    fp=results[(results.target == 0) & (results.y_predict == 1)]
    fn=results[(results.target == 1) & (results.y_predict == 0)]
    tp=results[(results.target == 1) & (results.y_predict == 1)]

    return (results, tn, fp, fn, tp)





# def get_feature_importance10(feature_importance, feature_names, y_train):
#     # Get the index of label=1 and label=0 in the y_resampled array
#     idx_label_1 = np.where(y_train == 1)[0]
#     idx_label_0 = np.where(y_train == 0)[0]


#     # Get the coefficients for label=1 and label=0
#     coefs_label_1 = feature_importance[idx_label_1]
#     coefs_label_0 = feature_importance[idx_label_0]

#     # Get the feature names for label=1 and label=0
#     feature_names_label_1 = [feature_names[i] for i in idx_label_1]
#     feature_names_label_0 = [feature_names[i] for i in idx_label_0]

#     # Create a dictionary of feature names and their corresponding coefficients for label=1 and label=0
#     coef_dict_label_1 = dict(zip(feature_names_label_1, coefs_label_1))
#     coef_dict_label_0 = dict(zip(feature_names_label_0, coefs_label_0))

#     # label=1
#     df_feature_importance1 = pd.DataFrame.from_dict(coef_dict_label_1, orient='index', columns=['importance'])
#     df_feature_importance1 = df_feature_importance1.reset_index()

#     df_feature_importance1 = df_feature_importance1.rename({'index':'feature'}, axis=1)
#     df_feature_importance1['abs_importance'] = abs(df_feature_importance1['importance'])
#     df_feature_importance1 = df_feature_importance1.sort_values(by='abs_importance', ascending=False).drop(columns='abs_importance')

#     # label=0
#     df_feature_importance0 = pd.DataFrame.from_dict(coef_dict_label_0, orient='index', columns=['importance'])
#     df_feature_importance0 = df_feature_importance0.reset_index()

#     df_feature_importance0 = df_feature_importance0.rename({'index':'feature'}, axis=1)
#     df_feature_importance0['abs_importance'] = abs(df_feature_importance0['importance'])
#     df_feature_importance0 = df_feature_importance0.sort_values(by='abs_importance', ascending=False).drop(columns='abs_importance')

#     return(df_feature_importance1, df_feature_importance0)