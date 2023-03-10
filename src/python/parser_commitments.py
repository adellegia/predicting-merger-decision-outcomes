# Data manipulation
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile

# Webscraping
import glob
import re
import requests
from bs4 import BeautifulSoup
import time
import datetime
from pandas.core.common import flatten
import os
from itertools import chain
from tqdm import tqdm
import json
import urllib.request
import lxml
from pdfminer.high_level import extract_text
import pdfplumber
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import PyPDF2

#--------------------------------------------------------------------------------------------------------------
# Function to parse pdfs
# def parse_pdf(pdf_list):

#     df_text = pd.DataFrame(columns = ['article', 'case_num', 'filename', 'text', 'lang'])

#     for pdf_file in tqdm(pdf_list):
#             article = re.search("art[0-9]+\.[0-9]+", pdf_file).group()
#             case_num = re.search("M\.[0-9]{4,5}", pdf_file).group()

#             regex = re.compile(r'(?<=/)[^/]+(?=\.pdf)')
#             match = regex.search(pdf_file)
#             filename = match.group()
            
#             try:
#                 with pdfplumber.open(pdf_file) as pdf:
#                     text = []
#                     for i in range(len(pdf.pages)):
#                         page = pdf.pages[i]
#                         page_text = page.extract_text()
#                         text.append(page_text)
#                     text = ' '.join(text)
#                     lang = detect(text)
#                 row = pd.DataFrame({'article': article, 'case_num': case_num,'filename': filename,
#                                         'text': text, 'lang': lang}, index=[0])
#                 df_text = pd.concat([row,df_text.loc[:]]).reset_index(drop=True)
#             except (IOError, FileNotFoundError) as ex:
#                 print("Skipping corrupted PDF file: ", ex)
#                 pass
#             except LangDetectException as ex:
#                 print("Skipping file with LangDetectException: ", ex)
#                 pass          
#     return df_text

#--------------------------------------------------------------------------------------------------------------
# Function to regex article number
def _article_match(txt):
    match = re.search(r"(?i)article\s*\d+\s*(\([^\)]+\))?\s*(\([^\)]+\))?", txt) 
    if match:
        first_match = match.group()
        return first_match.replace(" ", "").replace("\t", "").replace("\n", "").lower()
    else:
        return "None"

def _article62(txt):
    match = re.search(r"(?i)IN\s+CONJUNCTION\s+WITH\s+ART(?:ICLE)?\s+\d+\(\d+\)", txt)  
    if match:
        first_match = match.group()
        return first_match.replace(" ", "").replace("\t", "").replace("\n", "").lower()
    else:
        return "None"
    
#--------------------------------------------------------------------------------------------------------------
# Function to get date
def _extract_date_text(section_text):
    pattern = re.compile(r'Date.*?\n', re.IGNORECASE)
    match = pattern.search(section_text)
    if match:
        return match.group().strip()
    else:
        return ''

# Function to get year
def _extract_year_text(text):
    # Define a regular expression pattern to match years between 2004 and 2022
    pattern = r'\b([2][0][0-2][0-9])\b' #[2][0][0-2][0-9]

    # Use re.search() to extract the first match of the pattern in the subheading
    match = re.search(pattern, text)

    # If a match is found, extract the matched string, otherwise return None
    year = match.group(0) if match else None

    return year

#--------------------------------------------------------------------------------------------------------------
# Function to get subheadings and long_text_list
def _parse_subheadings_text(pdf_file):

    with pdfplumber.open(pdf_file) as pdf:
        text = []
        subheading = []
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            page_text = page.extract_text()
            text.append(page_text)

            if i == 0:
                article_txt = _article_match(page_text)
                article_62 = _article62(page_text)

            bold_text = page.filter(lambda obj: obj["object_type"] == "char" and "Bold" in obj["fontname"])
            bold_text = bold_text.extract_text()
            capitalized_text = [line.strip() for line in bold_text.split('\n') if line.isupper()]
            subheading.append(capitalized_text)

        long_text = ' '.join(text)
        long_text_list = [text.strip() for text in long_text.split('\n') if text.strip()]

        date = _extract_date_text(long_text)
        year = _extract_year_text(date)

        len_pdf = len(pdf.pages)

        #subheading = [x.strip() for x in subheading for x in x.split('\n') if  any(char.isalpha() for char in x.strip())]
        subheading_list = []
        for lst in subheading:
            subheading_list.extend(lst)

        prefixes = ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.", 
                    "I ", "II ", "III ", "IV ", "V ", "VI ", "VII ", "VIII ", "IX ", "X ", "XI ", "XII ", 
                    "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.",
                    "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "10 ", "11 ", "12 ",
                    "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "(10)", "(11)", "(12)"]
        retained_subheadings = []

        for subheading in subheading_list:
            for prefix in prefixes:
                if subheading.startswith(prefix):
                    retained_subheadings.append(subheading)
                    break

        if len_pdf<=5:
            bsn_act = long_text
        else:
            bsn_act = ""

    return([retained_subheadings, long_text_list, article_txt, article_62, 
            date, year, len_pdf, bsn_act, subheading_list])


#--------------------------------------------------------------------------------------------------------------
def _drop_empty_lists(d):
    return {k: v for k, v in d.items() if len(v)==1}

# Function to create dictionary of subheading key and index
def _find_indices(subheading, long_text_list):
    indices = {}
    for elem in subheading:
        indices[elem] = [i for i, x in enumerate(long_text_list) if x and x.find(elem) != -1]
    return _drop_empty_lists(indices)

#--------------------------------------------------------------------------------------------------------------
# Function to extract between subheading keys
def _extract_between_text(subheading_dict, long_text_list):

    # Initialize start and end variables to extract text between keys
    start_index = 0
    result = []

    # Iterate over the keys in the dictionary and extract the text between them
    for key in sorted(subheading_dict, key=subheading_dict.get):
        end_index = (subheading_dict[key])[0]
        # print(end_index[0])
        result.append(long_text_list[start_index:end_index])
        start_index = end_index

    # Extract the final text after the last key
    result.append(long_text_list[start_index:])

    return(result)

#--------------------------------------------------------------------------------------------------------------
# Function to get business activities for Simplified Procedure (if len(pdf.pages)<=3)
# def _parse_simplified_text(text):

#     # Define the start and end patterns
#     start_pattern = re.compile(r'(?i)business activities of the undertakings concerned are')
#     end_pattern = re.compile(r'(?i)after examination of the notification')

#     # Find the start and end positions of the patterns
#     start_pos = start_pattern.search(text).end()
#     end_pos = end_pattern.search(text).start()

#     # Extract the text between the patterns
#     result = text[start_pos:end_pos].strip()

#     return(result)

def _parse_simplified_text2(text):
    # Define the start and end patterns
    start_pattern = re.compile(r'(?i)business activities of the undertakings concerned are|business activities')
    end_pattern = re.compile(r'(?i)after examination of the notification|\n \n|\n\n')

    result = "None"
    try:
        # Find the start and end positions of the patterns
        start_pos = start_pattern.search(text).end()
        end_match = end_pattern.search(text[start_pos:])
        if end_match is None:
            end_pos = len(text)
        else:
            end_pos = start_pos + end_match.start()

        # Extract the text between the patterns
        result = text[start_pos:end_pos].strip()
    except AttributeError:
        pass
    except UnboundLocalError:
        pass

    return result

#--------------------------------------------------------------------------------------------------------------
# Function to parse pdf by section
def parse_pdf_section(pdf_list):

    df_text = pd.DataFrame(columns = ['date', 'year', 'len_pdf', 'article', 'article_txt', 'article_62', 'article_new', 
                                      'case_num', 'filename', 'file', 'section_text', 'bsn_act', 'simp_text'])
    simp_text="None"
    for pdf_file in tqdm(pdf_list):
            article = re.search("art[0-9]+\.[0-9]+", pdf_file).group()
            case_num = re.search("M\.[0-9]{4,5}", pdf_file).group()

            regex = re.compile(r'(?<=/)[^/]+(?=\.pdf)')
            match = regex.search(pdf_file)
            filename = match.group()

            regex2 = re.compile(r"\\([^\\]*)$")
            match2 = regex2.search(filename)
            file = match2.group()
            
            try:
                subheadings_text = _parse_subheadings_text(pdf_file)
                subheading = subheadings_text[0]
                long_text_list = subheadings_text[1]

                article_txt = subheadings_text[2]
                article_62 = subheadings_text[3]
                date = subheadings_text[4]
                year = subheadings_text[5]
                len_pdf = subheadings_text[6]
                bsn_act = subheadings_text[7]

                # change to 6.2 with in conjunction
                if article_62 in ['inconjunctionwitharticle6(2)', 'inconjunctionwithart6(2)']:
                    article_new = 'article6(2)'
                elif article_txt in ["article4(4)", "article22(3)", "article22", "article9(3)", "article9(3)(b)"]:
                    article_new = 'referral'
                else:
                    article_new = article_txt

                # parse according to simplified procedure
                if len_pdf > 5:
                    subheading_dict = _find_indices(subheading, long_text_list)

                    section_text = _extract_between_text(subheading_dict, long_text_list)
                    simp_text = "None"
                    row = pd.DataFrame({'date': date, 'year': year, 'len_pdf':len_pdf,
                                                'article': article, 'article_txt': article_txt, 'article_62': article_62, 'article_new': article_new,
                                                'case_num': case_num,'filename': filename, 'file': file,
                                                'section_text': [section_text], 'bsn_act': bsn_act, 'simp_text': simp_text}, index=[0])
                    df_text = pd.concat([row,df_text.loc[:]]).reset_index(drop=True)
                else:
                    section_text = "None"
                    try:
                        simp_text = _parse_simplified_text2(bsn_act)
                    except AttributeError:
                        pass

                    row = pd.DataFrame({'date': date, 'year': year, 'len_pdf':len_pdf,
                            'article': article, 'article_txt': article_txt, 'article_62': article_62, 'article_new': article_new,
                            'case_num': case_num,'filename': filename, 'file': file,
                            'section_text': [section_text], 'bsn_act': bsn_act, 'simp_text': simp_text}, index=[0])
                    df_text = pd.concat([row,df_text.loc[:]]).reset_index(drop=True)

            except (IOError, FileNotFoundError) as ex:
                print("Skipping corrupted PDF file: ", ex)
                pass        
    return df_text

#--------------------------------------------------------------------------------------------------------------
# Function to split parsed text and get section
def split_text(df):
    df_split = df.explode('section_text').reset_index(drop=True)

    # get subheading from section_text
    mask_non = df_split['simp_text'] == 'None'
    df_split.loc[mask_non, 'section'] = df_split.loc[mask_non, 'section_text'].str.get(0)

    # remove the first element/subheading of section_text
    remove_first = lambda x: x[1:]
    df_split.loc[mask_non,'section_text'] = df_split.loc[mask_non,'section_text'].apply(remove_first)

    return(df_split)

#--------------------------------------------------------------------------------------------------------------
# Function to clean parsed text
def clean_text(df_split):
    #remove if section_text is empty or len=1 but not simp_text = None
    df_split = df_split[(df_split['simp_text'] != "None") | ((df_split['section_text'].str.len() > 1) & (~df_split['section_text'].isnull()))]

    # remove based on subheading/section
    keep_mask = ~df_split['section'].str.lower().isin(['en', 'merger procedure', 'commission decision', 'european commission'])
    df_split = df_split[keep_mask].reset_index(drop=True)

    # # remove based on section_text list
    # keep_mask = ~(df_split['section_text'].apply(lambda x: any(isinstance(x, list) and 
    #                                                            any(s.lower() in t.lower() for t in x) 
    #                                                            for s in ['Only the English text is authentic', 'Only the [English] text is authentic', 'Only the English text is available', 'Commission Decision', 
    #                                                                      'Having regard to the Treaty establishing the European Community,', 'European Commission received a notification', 'European Commission received notification' ])))
    # df_split = df_split[keep_mask].reset_index(drop=True)

    return(df_split)

# #--------------------------------------------------------------------------------------------------------------
# # Function to remove CONCLUSION section
# def remove_conclusion(df_clean):
#     # remove conclusion
#     keep_mask = ~(df_clean['section'].str.lower().str.contains('conclusion'))

#     # Filter the DataFrame to keep only rows where the mask is True
#     df_filtered = df_clean[keep_mask].reset_index(drop=True)

#     return(df_filtered)

# #--------------------------------------------------------------------------------------------------------------
# # Function to remove COMMITMENTS section
# def remove_commitments(df_clean):
#     # remove commitments
#     keep_mask = ~(df_clean['section'].str.lower().str.contains('commitment|remedies|remedy|obligation|condition'))

#     # Filter the DataFrame to keep only rows where the mask is True
#     df_filtered = df_clean[keep_mask].reset_index(drop=True)

#     return(df_filtered)

#--------------------------------------------------------------------------------------------------------------
# Run all functions
def clean_parsed_pdf_section(df):
    df_split = split_text(df)
    df_clean = clean_text(df_split)

    # df_sub = df_clean[df_clean['simp_text'] == "None"]
    # df_simp = df_clean[df_clean['simp_text'] != "None"]

    # df_filtered=filter_conclusion(df_sub)
    # df_final_clean = pd.concat([df_filtered, df_simp]).reset_index(drop=True)

    df_final_clean = df_clean.reset_index(drop=True)
    return(df_final_clean)






















#--------------------------------------------------------------------------------------------------------------
# # Function to remove conclusion, and texts after that
# def _filter_group(group):
#     # Find the index of the row where 'section' contains 'conclusion' (case-insensitive)
#     conclusion_idx = group['section'].str.lower().str.contains('conclusion').idxmax()

#     # Create a boolean mask for rows to keep
#     keep_mask = group.index <= conclusion_idx

#     # Filter the group to keep only rows where the mask is True
#     filtered_group = group[keep_mask].reset_index(drop=True)

#     # Return the filtered group
#     return filtered_group

# def filter_conclusion(df_split):
#     # Group the DataFrame by 'case_num' and apply the filter_group function to each group
#     df_filtered = df_split.groupby('case_num', group_keys=False).apply(_filter_group).reset_index(drop=True)

#     # Create a boolean mask for rows to keep
#     keep_mask = ~(df_filtered['section'].str.lower().str.contains('conclusion'))

#     # Filter the DataFrame to keep only rows where the mask is True
#     df_filtered = df_filtered[keep_mask].reset_index(drop=True)
    
#     return(df_filtered)

