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
# Function to create folder
def _createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


#--------------------------------------------------------------------------------------------------------------
# Function to get list of merger decision links
def get_merger_links(decision_type):

    with open(f'../../../data/raw/' + str(decision_type), 'rb') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, features="lxml")

    decision_list = soup.find_all('td', attrs={'class': 'case'})

    case_details_list = []

    for i in decision_list:
        case_detail = i.find('a')
        case_num = case_detail.text.strip()
        case_details_list.append([case_num, case_detail['href']])   
    
    return case_details_list

#--------------------------------------------------------------------------------------------------------------
# Function to download EN pdf in case folder
def download_pdf(decision_page, article):
    url = 'https://ec.europa.eu/competition/elojade/isef/' + str(decision_page)

    page = requests.get(str(url))
    time.sleep(2)

    soup = BeautifulSoup(page.content, 'html.parser')
    
    # case number
    case_details = soup.find('div', attrs={'id': 'BodyContent'}).find('strong').text.strip()
    case_num = case_details[:case_details.find("\n")-1]

    # create case folder
    folder = f"../../../data/pdfs/{article}/{case_num}"
    _createFolder(folder)

    # download EN pdf/s
    table_decisions = soup.find('table', attrs={'id': 'decisions'})

    #links = table_decisions.find_all('a')
    links_tag = table_decisions.find_all('a', class_='ClassLink')

    link_list = []
    for a_tag in links_tag:
        if "en" in a_tag.text.lower():
            link_list.append(a_tag.get('href', []))

    i=0 # case file counter
    #j=0 # downloaded
    k=0 # unsuccessful
    for url in link_list:
        #TODO: if url empty, no pdf to download
        if url is None:
            print("No file to download " + case_num)
        else:
            i += 1
            print("Downloading {} file: ".format(case_num), i) 

            #filename=url[url.find("decisions/")+10:] # change to between / and .pdf
            
            regex = re.compile(r'(?<=/)[^/]+(?=\.pdf)')
            match = regex.search(url)
            filename = match.group()

            pdf_file = folder + "/"+ str(filename) + ".pdf"

            if os.path.exists(pdf_file):
                os.remove(pdf_file)

            try:
                urllib.request.urlretrieve(url, pdf_file)
                print("Downloaded {} file: ".format(case_num), i)
            except FileNotFoundError:
                print("File not found.")
                k += 1
                pass
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error: {e}")
                k += 1
                pass  
            except requests.exceptions.ContentTooShortError as e:
                print(f"ContentTooShortError: {e}")
                k += 1
                pass
    j=i-k
    return j,k


#--------------------------------------------------------------------------------------------------------------
# Function to download EN pdf in case folder
def download_pdf6(decision_page, article):
    url = 'https://ec.europa.eu/competition/elojade/isef/' + str(decision_page)

    page = requests.get(str(url))
    time.sleep(2)
    # except requests.exceptions.RemoteDisconnected:
    #     print("Remote disconnected, sleeping...")
    #     time.sleep(10)

    soup = BeautifulSoup(page.content, 'html.parser')
    
    # case number
    case_details = soup.find('div', attrs={'id': 'BodyContent'}).find('strong').text.strip()
    case_num = case_details[:case_details.find("\n")-1]

    # create case folder
    folder = f"../../../data/pdfs/{article}/{case_num}"
    _createFolder(folder)

    # download EN pdf/s
    table_decisions = soup.find('table', attrs={'id': 'decisions'})

    #links = table_decisions.find_all('a')
    links_tag = table_decisions.find_all('a', class_='ClassLink')

    link_list = []
    for a_tag in links_tag:
        if "en" in a_tag.text.lower():
            link_list.append(a_tag.get('href', []))

    i=0 # case file counter
    #j=0 # downloaded
    k=0 # unsuccessful
    for url in link_list:
        #TODO: if url empty, no pdf to download
        if url is None:
            print("No file to download " + case_num)
        else:
            i += 1
            #print("Downloading {} file: ".format(case_num), i) 

            #filename=url[url.find("decisions/")+10:] # change to between / and .pdf
            
            regex = re.compile(r'(?<=/)[^/]+(?=\.pdf)')
            match = regex.search(url)
            filename = match.group()

            pdf_file = folder + "/"+ str(filename) + ".pdf"

            if os.path.exists(pdf_file):
                os.remove(pdf_file)

            try:
                urllib.request.urlretrieve(url, pdf_file)
                #print("Downloaded {} file: ".format(case_num), i)
            except FileNotFoundError:
                #print("File not found.")
                k += 1
                pass
            except requests.exceptions.HTTPError as e:
                #print(f"HTTP Error: {e}")
                k += 1
                pass  
            except requests.exceptions.ContentTooShortError as e:
                #print(f"ContentTooShortError: {e}")
                k += 1
                pass
    j=i-k
    return j,k

#--------------------------------------------------------------------------------------------------------------
# Function to retrieve merger decision info
def get_merger_info(decision_page):

    url = 'https://ec.europa.eu/competition/elojade/isef/' + str(decision_page)

    page = requests.get(str(url))
    time.sleep(2)

    soup = BeautifulSoup(page.content, 'html.parser')
    
    # case number
    case_details = soup.find('div', attrs={'id': 'BodyContent'}).find('strong').text.strip()
    case_num = case_details[:case_details.find("\n")-1]

    # retrive merger details
    table = soup.find('table', attrs={'class': 'details'})

    table_detail = table.find_all('td', attrs={'class': 'ClassTextDetail'})

    # notification date
    notification_date = table_detail[0].text.strip()

    # provisional deadline  TODO: find first date format
    prov_info = table_detail[1].text.strip()
    #prov_deadline = prov_info[:10]
    #prov_details = prov_info[prov_info.find("\t\n")+2:]

    # Prior publication in Official Journal
    OJ_deadline = table_detail[2].text.strip() # find all 'a' convert to list

    # Concerns economic activity (NACE)
    nace_info = table_detail[3].text.strip()
    #nace = nace_info[:nace_info.find("\n")]
    #nace_details = nace_info[nace_info.find(" ")+2:]

    # regulation
    regulation = table_detail[4].text.strip()

    decision_info = [case_num, notification_date, prov_info, OJ_deadline, nace_info, regulation]
        
    return decision_info


#--------------------------------------------------------------------------------------------------------------
# Function to parse pdfs
def parse_pdf(pdf_list):

    df_text = pd.DataFrame(columns = ['article', 'case_num', 'filename', 'text', 'lang'])

    for pdf_file in tqdm(pdf_list):
            article = re.search("art[0-9]+\.[0-9]+", pdf_file).group()
            case_num = re.search("M\.[0-9]{4,5}", pdf_file).group()

            regex = re.compile(r'(?<=/)[^/]+(?=\.pdf)')
            match = regex.search(pdf_file)
            filename = match.group()
            
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    text = []
                    for i in range(len(pdf.pages)):
                        page = pdf.pages[i]
                        page_text = page.extract_text()
                        text.append(page_text)
                    text = ' '.join(text)
                    lang = detect(text)
                row = pd.DataFrame({'article': article, 'case_num': case_num,'filename': filename,
                                        'text': text, 'lang': lang}, index=[0])
                df_text = pd.concat([row,df_text.loc[:]]).reset_index(drop=True)
            except (IOError, FileNotFoundError) as ex:
                print("Skipping corrupted PDF file: ", ex)
                pass
            except LangDetectException as ex:
                print("Skipping file with LangDetectException: ", ex)
                pass          
    return df_text

#--------------------------------------------------------------------------------------------------------------
# Function to regex article number
def article_match(txt):
    match = re.search(r"(?i)article\s*\d+\s*(\([^\)]+\))?\s*(\([^\)]+\))?", txt) 
    if match:
        first_match = match.group()
        return first_match.replace(" ", "").replace("\t", "").replace("\n", "").lower()
    else:
        return "None"

def article62(txt):
    match = re.search(r"(?i)IN\s+CONJUNCTION\s+WITH\s+ART(?:ICLE)?\s+\d+\(\d+\)", txt)  
    if match:
        first_match = match.group()
        return first_match.replace(" ", "").replace("\t", "").replace("\n", "").lower()
    else:
        return "None"

#--------------------------------------------------------------------------------------------------------------
#


# from langdetect.lang_detect_exception import LangDetectException
# import PyPDF2

# def parse_pdf(pdf_list):

#     df_text = pd.DataFrame(columns = ['article', 'case_num', 'filename', 'text', 'lang'])

#     for pdf_file in tqdm(pdf_list):
#             article = re.search("art[0-9]+\.[0-9]+", pdf_file).group()
#             case_num = re.search("M\.[0-9]{4,5}", pdf_file).group()

#             regex = re.compile(r'(?<=/)[^/]+(?=\.pdf)')
#             match = regex.search(pdf_file)
#             filename = match.group()
            
#             try:
#                 with open(pdf_file, "rb") as f:
#                     pdf_reader = PyPDF2.PdfReader(f)
#                     if not pdf_reader.is_encrypted:
#                         with pdfplumber.open(pdf_file) as pdf:
#                             text = []
#                             for i in range(len(pdf.pages)):
#                                 page = pdf.pages[i]
#                                 page_text = page.extract_text()
#                                 text.append(page_text)
#                             text = ' '.join(text)
#                             lang = detect(text)
#                         row = pd.DataFrame({'article': article, 'case_num': case_num,'filename': filename,
#                                                 'text': text, 'lang': lang}, index=[0])
#                         df_text = pd.concat([row,df_text.loc[:]]).reset_index(drop=True)
#             except PyPDF2.utils.PdfReadError as e:
#                 print("Skipping corrupted PDF file: ", e)

#             # try:
#             #     with pdfplumber.open(pdf_file) as pdf:
#             #         text = []
#             #         for i in range(len(pdf.pages)):
#             #             page = pdf.pages[i]
#             #             page_text = page.extract_text()
#             #             text.append(page_text)
#             #         text = ' '.join(text)
#             #         lang = detect(text)
#             #     row = pd.DataFrame({'article': article, 'case_num': case_num,'filename': filename,
#             #                             'text': text, 'lang': lang}, index=[0])
#             #     df_text = pd.concat([row,df_text.loc[:]]).reset_index(drop=True)
#             except (IOError, FileNotFoundError) as ex:
#                 print("Skipping corrupted PDF file: ", ex)
#                 pass
#             except LangDetectException as ex:
#                 print("Skipping file with LangDetectException: ", ex)
#                 pass          
#     return df_text


# ContentTooShortError, TimeoutError, HTTPError, ConnectionError, RemoteDisconnected 
