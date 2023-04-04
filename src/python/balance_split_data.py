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
# function to retain section
def retain_section(df, section):
    df_section = df[df['section_fin']==section]
    return df_section

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

    df_unique = df1.groupby(['label', 'case_num','article']).first().reset_index()[['label', 'case_num', 'article']]

    print("Total decisions:", len(df_unique.index))
    print(df_unique['label'].value_counts())
    # print(df_unique['article'].value_counts())

    return(df_unique)

########################################################################################################
# function to create balanced dataset and excluded dataset
def create_balanced_excluded(df_unique, df1, random_seed=42):

    decision_id_rus, y_rus = _balance(df_unique['case_num'], df_unique['label'], random_seed=random_seed)
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
def create_train_test_excluded(df_balanced, df_balanced_unique, df_excluded, random_state=42):

    # Split train and test based on case_num randomly, stratify by label and article
    case_label = df_balanced_unique['label']
    casenum = df_balanced_unique['case_num']
    y = df_balanced_unique[['label', 'article']]

    case_num_train, case_num_test, case_label_train, case_label_test = train_test_split(casenum, case_label, test_size=0.2, stratify=y, random_state=random_state)

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
def group_by_case(df_train, df_test, df_test1, df_excluded, text='text_clean'):
    # Group df_train by 'case_num', 'article', 'file', and join the text column
    
    df_train_grouped = df_train.groupby(['year', 'case_num', 'article', 'label'])[text].agg(' '.join).reset_index() #'file',
    print("Grouping df_train by case_num",
          "1:", len(df_train_grouped[df_train_grouped['label'] == 1]['case_num'].unique()),
          "0:", len(df_train_grouped[df_train_grouped['label'] == 0]['case_num'].unique()))

    # Group df_test by 'case_num', 'article', 'file', and join the text column
    df_test_grouped = df_test.groupby(['year', 'article', 'case_num', 'label'])[text].agg(' '.join).reset_index()
    print("Grouping df_test by case_num", 
          "1:", len(df_test_grouped[df_test_grouped['label'] == 1]['case_num'].unique()),
          "0:", len(df_test_grouped[df_test_grouped['label'] == 0]['case_num'].unique()))


    df_test1_grouped = df_test1.groupby(['year', 'article', 'case_num', 'label'])[text].agg(' '.join).reset_index()
    print("Grouping df_test1 by case_num", 
          "1:", len(df_test1_grouped[df_test1_grouped['label'] == 1]['case_num'].unique()),
          "0:", len(df_test1_grouped[df_test1_grouped['label'] == 0]['case_num'].unique()))

    # Group df_excluded by 'case_num', 'article', 'file', and join the text column
    df_excluded_grouped = df_excluded.groupby(['year', 'article', 'case_num', 'label'])[text].agg(' '.join).reset_index()
    print("Grouping df_excluded by case_num with len:", df_excluded['case_num'].nunique())

    return(df_train_grouped, df_test_grouped, df_test1_grouped, df_excluded_grouped)

########################################################################################################
# function to create features and labels
def create_feature_label(df_train_grouped, df_test_grouped, df_test1_grouped, df_excluded_grouped, text='text_clean'):
    # Train set
    X_train = df_train_grouped[text]
    y_train = df_train_grouped['label']

    # Combined test set
    X_test = df_test_grouped[text]
    y_test = df_test_grouped['label']

    # 20% of full data
    X_test1 = df_test1_grouped[text]
    y_test1 = df_test1_grouped['label']

    # excluded only
    X_test2 = df_excluded_grouped[text]
    y_test2 = df_excluded_grouped['label']

    return(X_train, y_train, X_test, y_test, X_test1, y_test1, X_test2, y_test2)
#------------------------------------------------------------------------------------------------------------------