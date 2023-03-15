# Data visualization
import matplotlib.pyplot as plt

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
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
    df1['label'] = le.fit_transform(df1['phase2'])

    return(df1)

########################################################################################################
# function to balance data
def _balance(decision_id, Ytrain, random_seed=42):
    print('Balancing...')
    v = [i for i, val in enumerate(Ytrain) if val == 1]
    nv = [i for i, val in enumerate(Ytrain) if val == 0]

    if len(nv) < len(v):
        v = random.sample(v, len(nv))
    else:
        nv = random.sample(nv, len(v))

    indices = v + nv
    random.seed(random_seed)  # Set the random seed
    random.shuffle(indices)

    decision_id = [decision_id[i] for i in indices]
    Ytrain = [Ytrain[i] for i in indices]

    print("Total decisions:", len(decision_id))
    print("Labels distribution:", "\n", (pd.DataFrame(Ytrain)[0].value_counts()))
    return decision_id, Ytrain

########################################################################################################
# function to balance data_unique cases
def balance_unique_id(data):
    df1 = data.copy()

    df_unique = df1.groupby(['label', 'case_num','article_new']).first().reset_index()[['label', 'case_num', 'article_new']]

    print("Total decisions:", len(df_unique.index))
    print(df_unique['label'].value_counts())
    # print(df_unique['article_new'].value_counts())

    return(df_unique)

########################################################################################################
# function to create balanced dataset and excluded dataset
def create_balanced_excluded(df_unique, df1):

    decision_id_rus, y_rus = _balance(df_unique['case_num'], df_unique['label'], random_seed=42)
    df_balanced = df1[df1['case_num'].isin(decision_id_rus) & df1['label'].isin(y_rus)].reset_index(drop=True)
    df_balanced.groupby('label')['case_num'].nunique()

    df_balanced_unique = df_unique[df_unique['case_num'].isin(decision_id_rus) & df_unique['label'].isin(y_rus)].reset_index(drop=True)
    df_balanced_unique.groupby('label')['case_num'].nunique()

    df_excluded = df1[~(df1['case_num'].isin(decision_id_rus) & df1['label'].isin(y_rus))]
    #print(df_excluded.groupby('label')['case_num'].nunique(), "\n",
    #df_excluded['label'].value_counts())

    return(df_balanced_unique, df_balanced, df_excluded)

########################################################################################################
# function to split train and test
def create_train_test_excluded(df_balanced, df_balanced_unique, df_excluded):

    # Split train and test based on case_num randomly, stratify by label and article
    case_label = df_balanced_unique['label']
    casenum = df_balanced_unique['case_num']
    y = df_balanced_unique[['label', 'article_new']]

    case_num_train, case_num_test, case_label_train, case_label_test = train_test_split(casenum, case_label, test_size=0.2, stratify=y, random_state=42)

    # Print the shape of each set to verify that the data has been split correctly
    print("Training set shape:", case_label_train.shape, case_label_train.shape)
    print("Test set shape:", case_num_test.shape, case_label_test.shape)

    # Create train_set from df_balanced
    df_train = df_balanced[df_balanced['case_num'].isin(case_num_train) & df_balanced['label'].isin(case_label_train)].reset_index(drop=True)
    print("Creating df_train", 
          "1:", len(df_train[df_train['label'] == 1]['case_num'].unique()),
          "0:", len(df_train[df_train['label'] == 0]['case_num'].unique()))
    
    # Create test_set from df_balanced
    df_test1 = df_balanced[df_balanced['case_num'].isin(case_num_test) & df_balanced['label'].isin(case_label_test)].reset_index(drop=True)
    print("Creating df_test1", 
          "1:", len(df_test1[df_test1['label'] == 1]['case_num'].unique()),
          "0:", len(df_test1[df_test1['label'] == 0]['case_num'].unique()))
    

    # Create df_test to add excluded cases to df_test1
    print("Creating df_test concatenated with df_excluded with len:", df_excluded['case_num'].nunique())
    df_test = pd.concat([df_test1, df_excluded], axis=0)
    df_test = df_test.reset_index(drop=True)
    print("Creating df_test", 
          "1:", len(df_test[df_test['label'] == 1]['case_num'].unique()),
          "0:", len(df_test[df_test['label'] == 0]['case_num'].unique()))

    return(df_train, df_test, df_test1)

########################################################################################################
# function to group by case_num train and test
def group_by_case(df_train, df_test, df_test1, df_excluded):
    # Group df_train by 'case_num', 'article_new', 'file', and join the 'text_clean' column
    
    df_train_grouped = df_train.groupby(['year', 'case_num', 'file', 'article_new', 'label'])['text_clean'].agg(' '.join).reset_index()
    print("Grouping df_train by case_num",
          "1:", len(df_train_grouped[df_train_grouped['label'] == 1]['case_num'].unique()),
          "0:", len(df_train_grouped[df_train_grouped['label'] == 0]['case_num'].unique()))

    # Group df_test by 'case_num', 'article_new', 'file', and join the 'text_clean' column
    df_test_grouped = df_test.groupby(['year', 'article_new', 'case_num', 'file', 'label'])['text_clean'].agg(' '.join).reset_index()
    print("Grouping df_test by case_num", 
          "1:", len(df_test_grouped[df_test_grouped['label'] == 1]['case_num'].unique()),
          "0:", len(df_test_grouped[df_test_grouped['label'] == 0]['case_num'].unique()))


    df_test1_grouped = df_test1.groupby(['year', 'article_new', 'case_num', 'file', 'label'])['text_clean'].agg(' '.join).reset_index()
    print("Grouping df_test1 by case_num", 
          "1:", len(df_test1_grouped[df_test1_grouped['label'] == 1]['case_num'].unique()),
          "0:", len(df_test1_grouped[df_test1_grouped['label'] == 0]['case_num'].unique()))

    # Group df_excluded by 'case_num', 'article_new', 'file', and join the 'text_clean' column
    df_excluded_grouped = df_excluded.groupby(['year', 'article_new', 'case_num', 'file', 'label'])['text_clean'].agg(' '.join).reset_index()
    print("Grouping df_excluded by case_num with len:", df_excluded['case_num'].nunique())

    return(df_train_grouped, df_test_grouped, df_test1_grouped, df_excluded_grouped)

########################################################################################################
# function to create features and labels
def create_feature_label(df_train_grouped, df_test_grouped, df_test1_grouped, df_excluded_grouped):
    # Train set
    X_train = df_train_grouped['text_clean']
    y_train = df_train_grouped['label']

    # Combined test set
    X_test = df_test_grouped['text_clean']
    y_test = df_test_grouped['label']

    # 20% of full data
    X_test1 = df_test1_grouped['text_clean']
    y_test1 = df_test1_grouped['label']

    # excluded only
    X_test2 = df_excluded_grouped['text_clean']
    y_test2 = df_excluded_grouped['label']

    return(X_train, y_train, X_test, y_test, X_test1, y_test1, X_test2, y_test2)
#------------------------------------------------------------------------------------------------------------------

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
    
    # Evaluate the performance of the model
    accuracy = accuracy_score(Ytest, Ypredict) *100.0
    precision = precision_score(Ytest, Ypredict, average='binary')
    recall = recall_score(Ytest, Ypredict, average='binary')
    f_score = 2 * (precision * recall) / (precision + recall)

    print(f' Accuracy: {accuracy:.2f} \n Precision: {precision:.3f} \n Recall: {recall:.3f} \n F1: {f_score:.3f}')


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

    # Sort the dataframe by importance and get the top 10 features
    top_features = df_features[:10]

    # Print the top 10 features and their importances
    # print(top_features)
    return(df_features)




















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
    pipeline.fit(Xtrain, Ytrain)  # fit the pipeline on the whole training data
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
    df_feature_importance = df_feature_importance.sort_values(by='abs_importance', ascending=False).drop(columns='abs_importance')

    # Print the feature importance values
    # print(df_feature_importance)
    
    return (df_feature_importance, feature_importance, feature_names)


#######################################################################################################
# create df of feature importances by label
def get_feature_importance10(feature_importance, feature_names, y_train):
    # Get the index of label=1 and label=0 in the y_resampled array
    idx_label_1 = np.where(y_train == 1)[0]
    idx_label_0 = np.where(y_train == 0)[0]


    # Get the coefficients for label=1 and label=0
    coefs_label_1 = feature_importance[idx_label_1]
    coefs_label_0 = feature_importance[idx_label_0]

    # Get the feature names for label=1 and label=0
    feature_names_label_1 = [feature_names[i] for i in idx_label_1]
    feature_names_label_0 = [feature_names[i] for i in idx_label_0]

    # Create a dictionary of feature names and their corresponding coefficients for label=1 and label=0
    coef_dict_label_1 = dict(zip(feature_names_label_1, coefs_label_1))
    coef_dict_label_0 = dict(zip(feature_names_label_0, coefs_label_0))

    # label=1
    df_feature_importance1 = pd.DataFrame.from_dict(coef_dict_label_1, orient='index', columns=['importance'])
    df_feature_importance1 = df_feature_importance1.reset_index()

    df_feature_importance1 = df_feature_importance1.rename({'index':'feature'}, axis=1)
    df_feature_importance1['abs_importance'] = abs(df_feature_importance1['importance'])
    df_feature_importance1 = df_feature_importance1.sort_values(by='abs_importance', ascending=False).drop(columns='abs_importance')

    # label=0
    df_feature_importance0 = pd.DataFrame.from_dict(coef_dict_label_0, orient='index', columns=['importance'])
    df_feature_importance0 = df_feature_importance0.reset_index()

    df_feature_importance0 = df_feature_importance0.rename({'index':'feature'}, axis=1)
    df_feature_importance0['abs_importance'] = abs(df_feature_importance0['importance'])
    df_feature_importance0 = df_feature_importance0.sort_values(by='abs_importance', ascending=False).drop(columns='abs_importance')

    return(df_feature_importance1, df_feature_importance0)