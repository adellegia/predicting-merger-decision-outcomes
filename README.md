# Predicting merger decision outcomes of the European Commission: A Natural Language Processing and Machine Learning approach

```
predicting-merger-decision-outcomes/
├───data
│   ├───decision_types                          - merger decision types by article
│   ├───parsed                                  - parsed datasets
│   ├───pdfs                                    - raw merger decision PDFs
│   ├───processed                               - pre-processed data
│   └───raw                                     - raw html of search tool results in .txt files
├───notebooks                                   
│   ├───data_preparation                        - webscraping, text parsing and pre-processing, data cleaning
│   ├───descriptives                            - descriptive statistics
│   ├───experiment_1                            - using full text from 4 sections (grid search CV, training CV and evaluation, feature importance, robustness checks)
│   ├───experiment_2                            - using single section text (training CV and evaluation, feature importance)
├───output
│   ├───figures                                 - plots and charts
│   ├───preds                                   - model predictions
│   └───tables                                  - tables of descriptive statistics
├───src
│   └───python                                  - functions for webscraping, parsing, pre-processing, balancing and splitting data, parameter tuning, model training and evaluation
├─.gitignore                                    - .gitignore files                          
├─environment.yml                               - project environment                     
├─README.md
└─Arbo_MaAdelleGia_master_thesis.pdf            - master thesis (digital copy)
```

## Executive Summary
Recent developments in data science provide new opportunities for analyzing large volumes of unstructured data, such as merger decision reports. While some studies have used Natural Language Processing (NLP) and Machine Learning (ML) techniques to forecast judicial decisions, text-based prediction of merger decision outcomes remains unexplored.

This thesis aims to explore the feasibility of applying NLP and ML to predict antitrust decisions under the EU Merger Regulation 2004 and understand the European Commission’s (EC) merger review process. By analyzing the language used in merger decision reports, as proxy for the facts and conditions of a proposed merger which would otherwise only be available in the confidential merger filings, this study seeks to uncover patterns and factors that contribute to the EC’s decision-making process. Given the limited review period and the numerous notifications received by the EC, building a text-based predictive model can aid merger review. The findings of this study have important implications for antitrust agencies, firms, and researchers interested in understanding the drivers of merger decisions.

Results show that a Support Vector Machine (SVM) linear classifier was the best-performing model in predicting whether a merger was approved with or without conditions and it was better at predicting cases ‘approved with conditions’ than cases ‘approved unconditionally.’ It achieved a high recall of 84% identifying cases with serious anticompetitive effects to prevent potential competition harm like higher prices, decreased quality of goods and services, and reduced innovation. Practical implementation will require a thorough cost-benefit analysis to determine the optimal trade-off between recall and precision, which ultimately depends on the competition authority’s objectives and priorities.

This thesis also highlights the model’s limitations and challenges in practical applications, including the need for expert judgment and human analysis. Future research can improve the study’s framework by collecting more data, considering the time variable when splitting the training and test sets, and exploring different machine learning models and more sophisticated deep learning algorithms.

Overall, this study opens up exciting possibilities for leveraging data science tools in antitrust decision-making as a supplement to expert judgment and analysis.

## Data
The pre-processed data used for modeling is available [here](https://www.dropbox.com/scl/fo/szosfc6m6w41n4bp19z7s/h?dl=0&rlkey=oit0pvre3o2j3qatyq709jhh0).

## Code
Code was self-written, debugged with StackOverflow and ChatGPT, unless noted otherwise. The code for the parameter tuning, model training with tenfold cross-validation, and model evaluation was adapted from Medvedeva et al. (2020) available [here](https://github.com/masha-medvedeva/ECtHR_crystal_ball)

## Author
Ma. Adelle Gia Arbo ([GitHub](https://github.com/adellegia), [LinkedIn](https://www.linkedin.com/in/ma-adelle-gia-arbo/))

## License
The material in this repository is made available under the [MIT license](http://opensource.org/licenses/mit-license.php). 
