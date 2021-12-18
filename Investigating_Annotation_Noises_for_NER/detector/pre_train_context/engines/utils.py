# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: utils.py
# @Project: sequence-lableing-vex

import re, logging, datetime, csv
import pandas as pd
import xlrd
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import numpy as np



def get_logger(log_dir):
    log_file = log_dir + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d: %H %M %S'))

    return logger


def get_test_entities_and_sentences(X_tokens):
    sentences = []
    entities = []
    for token_list in X_tokens:
        sentence = ' '.join(token_list)
        entity = ' '.join([word for word in token_list[10:14] if word != '<PAD>'])
        sentences.append(sentence)
        entities.append(entity)
    return sentences,entities

def classify_metrics(y_true,y_pred,measuring_metrics):
    y_true=[np.nonzero(y_id_list)[0][0] for y_id_list in y_true]
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred,average='macro')
    recall = recall_score(y_true,y_pred,average='macro')
    f1=f1_score(y_true,y_pred,average='macro')
    results = {}
    for measu in measuring_metrics:
        results[measu] = vars()[measu]
    return results

def read_excel(input_filename):
    excel_content = xlrd.open_workbook(input_filename,'rb').sheet_by_name('Sheet')
    sentences = excel_content.col_values(0)
    types = excel_content.col_values(1)
    return sentences,types





def save_csv_(df_, file_name, names, delimiter='t'):
    if delimiter == 't':
        sep = "\t"
    elif delimiter == 'b':
        sep = " "
    else:
        sep = delimiter

    df_.to_csv(file_name, quoting=csv.QUOTE_NONE,
               columns=names, sep=sep, header=False,
               index=False)
