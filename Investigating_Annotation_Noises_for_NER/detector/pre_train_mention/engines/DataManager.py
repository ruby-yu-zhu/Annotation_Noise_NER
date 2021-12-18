# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: DataManager.py
# @Project: sequence-lableing-vex

import os, logging
import numpy as np
from engines.utils import read_excel
import jieba,re
jieba.setLogLevel(logging.INFO)
from itertools import chain



class DataManager:
    def __init__(self, configs, logger):
        self.configs=configs
        self.train_file = configs.train_file
        self.logger = logger

        self.UNKNOWN = "<UNK>"
        self.PADDING = "<PAD>"

        self.train_file = configs.datasets_fold + "/" + configs.train_file
        self.test_file = configs.datasets_fold + "/" + configs.test_file


        self.label = configs.label

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim
        self.max_char_len = configs.max_char_len

        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = self.vocabs_dir + "/token2id"
        self.label2id_file = self.vocabs_dir + "/label2id"
        self.char2id_file = self.vocabs_dir + "/char2id"

        self.token2id, self.id2token, self.label2id, self.id2label, self.char2id, self.id2char = self.loadVocab()

        self.max_token_number = len(self.token2id)
        self.max_char_num = len(self.char2id)
        self.max_label_number = len(self.label2id)

        jieba.load_userdict(self.token2id.keys())

        self.logger.info("dataManager initialed...\n")

    def loadVocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info("vocab files not exist, building vocab...")
            return self.buildVocab(self.train_file)

        self.logger.info("loading vocab...")
        token2id = {}
        id2token = {}
        with open(self.token2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id = {}
        id2label = {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                label = row.split('\t')[0]
                label_id = int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        char2id = {}
        id2char = {}
        with open(self.char2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.rstrip()
                char = row.split('\t')[0]
                char_id = int(row.split('\t')[1])
                char2id[char] = char_id
                id2char[char_id] = char

        return token2id, id2token, label2id, id2label, char2id, id2char

    def buildVocab(self, train_path):
        train_sentences,train_types = read_excel(train_path)
        tokens = list(set(list(chain(*[re.split(' ',sentence.strip()) for sentence in train_sentences]))))
        labels = ['LOC','PER','ORG','MISC']
        char_dic = list(set(list(chain(*[[c for c in str] for str in tokens]))))
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        label2id = dict(zip(labels, range(1, len(labels))))
        char2id = dict(zip(char_dic,range(1, len(char_dic) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        id2label = dict(zip(range(1, len(labels)), labels))
        id2char = dict(zip(range(1,len(char_dic) + 1), char_dic))
        id2token[0] = self.PADDING
        id2char[0] = self.PADDING
        token2id[self.PADDING] = 0
        char2id[self.PADDING] = 0
        id2token[len(tokens) + 1] = self.UNKNOWN
        id2char[len(char_dic) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1
        char2id[self.UNKNOWN] = len(char_dic) + 1

        self.saveVocab(id2token, id2label, id2char)

        return token2id, id2token, label2id, id2label, char2id, id2char

    def saveVocab(self, id2token, id2label, id2char):
        with open(self.token2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + "\t" + str(idx) + "\n")
        with open(self.label2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + "\t" + str(idx) + "\n")
        with open(self.char2id_file, "w", encoding='utf-8') as outfile:
            for idx in id2char:
                outfile.write(id2char[idx] + "\t" + str(idx) + "\n")

    def getEmbedding(self, embed_file):
        print("begin to embedding!------------------")
        emb_matrix = np.random.normal(loc=0.0, scale=0.08, size=(len(self.token2id.keys()), self.embedding_dim))
        emb_matrix[self.token2id[self.PADDING], :] = np.zeros(shape=(self.embedding_dim))

        with open(embed_file, "r", encoding="utf-8") as infile:
            for row in infile:
                row = row.rstrip()
                items = row.split()
                token = items[0]
                assert self.embedding_dim == len(
                    items[1:]), "embedding dim must be consistent with the one in `token_emb_dir'."
                emb_vec = np.array([float(val) for val in items[1:]])
                if token in self.token2id.keys():
                    emb_matrix[self.token2id[token], :] = emb_vec
        print("the shape of embedding is:",emb_matrix.shape)

        return emb_matrix

    def token2ids_map(self,list):
        ids_list = []
        for word in list:
            if word not in self.token2id:
                ids_list.append(self.token2id[self.UNKNOWN])
            else:
                ids_list.append(self.token2id[word])
        return ids_list

    def char2ids_map(self,x):
        if x not in self.char2id:
            return self.char2id[self.UNKNOWN]
        else:
            return self.char2id[x]


    def tokens_mask(self,list):
        mask_list = []
        for word in list:
            if word == '<PAD>':
                mask_list.append(0)
            else:
                mask_list.append(1)
        return mask_list

    def process_char_padding(self,X_tokens):
        X_chars = []
        for token_list in X_tokens:
            char_list = []
            for token in token_list:
                if token != '<PAD>':
                    x_c = [self.char2ids_map(c) for c in token]
                    if len(x_c) >= self.max_char_len:
                        x_c = x_c[0:self.max_char_len]
                    else:
                        x_c = x_c + (self.max_char_len-len(x_c))*[self.char2id['<PAD>']]
                else:
                    x_c = self.max_char_len*[self.char2id['<PAD>']]
                char_list.append(x_c)
            X_chars.append(char_list)
        return X_chars

    def y_ids_to_matrixs(self,y_ids):
        y_ids_matrix = []
        for i in range(len(y_ids)):
            raw_matrixs = [0,0,0,0]
            raw_matrixs[y_ids[i]] = 1
            y_ids_matrix.append(raw_matrixs)
        return y_ids_matrix

    def getTrainingSet(self,train_val_ratio=0.8):
        sentences,types = read_excel(self.train_file)
        X_tokens = [re.split(' ',sentence.strip()) for sentence in sentences]
        X_mask = [self.tokens_mask(token_list) for token_list in X_tokens]#[sample_nums,max_seq_len]
        X_token_ids = [self.token2ids_map(token_list) for token_list in X_tokens]#[samples_nums,max_seq_len]
        y_ids = [self.label2id[label] for label in types]#[sample_nums]
        y_ids = self.y_ids_to_matrixs(y_ids)
        X_chars = self.process_char_padding(X_tokens)
        # for char_list in X_chars:
        #     if len(char_list) != 15:
        #         print('char wrong')
        #     for list in char_list:
        #         if len(list) != 50:
        #             print("char_nums wrong")

        X_tokens = np.array(X_tokens)
        X_mask = np.array(X_mask)  # [sample_nums,max_seq_len]
        X_token_ids = np.array(X_token_ids)  # [samples_nums,max_seq_len]
        y_ids = np.array(y_ids)  # [sample_nums]
        X_chars = np.array(X_chars)

        # shuffle the samples
        num_samples = len(X_tokens)
        indexs = np.arange(num_samples)
        np.random.shuffle(indexs)
        X_tokens = X_tokens[indexs]
        X_mask = X_mask[indexs]
        X_token_ids = X_token_ids[indexs]
        y_ids = y_ids[indexs]
        X_chars = X_chars[indexs]

        #pick_train_datas--------------------------------
        X_train_tokens = X_tokens[:int(num_samples * train_val_ratio)]
        X_train_token_ids = X_token_ids[:int(num_samples * train_val_ratio)]
        X_train_mask = X_mask[:int(num_samples * train_val_ratio)]
        X_train_chars = X_chars[:int(num_samples * train_val_ratio)]
        y_train_ids = y_ids[:int(num_samples * train_val_ratio)]

        #pick_dev_data-----------------------------------
        X_dev_tokens = X_tokens[int(num_samples * train_val_ratio):]
        X_dev_token_ids = X_token_ids[int(num_samples * train_val_ratio):]
        X_dev_mask = X_mask[int(num_samples * train_val_ratio):]
        X_dev_chars = X_chars[int(num_samples * train_val_ratio):]
        y_dev_ids = y_ids[int(num_samples * train_val_ratio):]

        self.logger.info("\ntraining set size: %d, validating set size: %d\n" % (len(X_train_tokens), len(X_dev_tokens)))
        # print(X_train_tokens.shape)
        # print(X_train_token_ids.shape)
        # print(X_train_mask.shape)
        # print(X_train_chars.shape)
        # print(y_train_ids.shape)
        return X_train_tokens,X_train_token_ids,X_train_mask,X_train_chars,y_train_ids,\
               X_dev_tokens,X_dev_token_ids,X_dev_mask,X_dev_chars,y_dev_ids

    def getTestingSet(self):
        sentences, types = read_excel(self.test_file)
        X_tokens = [re.split(' ', sentence.strip()) for sentence in sentences]
        X_mask = [self.tokens_mask(token_list) for token_list in X_tokens]  # [sample_nums,max_seq_len]
        X_token_ids = [self.token2ids_map(token_list) for token_list in X_tokens]  # [samples_nums,max_seq_len]
        y_ids = [self.label2id['PER'] for label in types]  # [sample_nums]
        y_ids = self.y_ids_to_matrixs(y_ids)
        X_chars = self.process_char_padding(X_tokens)
        X_tokens = np.array(X_tokens)
        X_mask = np.array(X_mask)  # [sample_nums,max_seq_len]
        X_token_ids = np.array(X_token_ids)  # [samples_nums,max_seq_len]
        y_ids = np.array(y_ids)  # [sample_nums]
        X_chars = np.array(X_chars)
        return X_tokens,X_token_ids,X_mask,X_chars,y_ids

    def nextBatch(self, X_tokens,X_token_ids,X_mask,X_chars,y_ids, start_index):
        last_index = start_index + self.batch_size
        X_batch = list(X_tokens[start_index:min(last_index, len(X_tokens))])
        X_ids_batch = list(X_token_ids[start_index:min(last_index, len(X_tokens))])
        X_mask_batch = list(X_mask[start_index:min(last_index, len(X_tokens))])
        X_chars_batch = list(X_chars[start_index:min(last_index, len(X_tokens))])
        y_ids_batch = list(y_ids[start_index:min(last_index, len(X_tokens))])
        if last_index > len(X_tokens):
            left_size = last_index - (len(X_tokens))
            for i in range(left_size):
                index = np.random.randint(len(X_tokens))
                X_batch.append(X_tokens[index])
                X_ids_batch.append(X_token_ids[index])
                X_mask_batch.append(X_mask[index])
                X_chars_batch.append(X_chars[index])
                y_ids_batch.append(y_ids[index])
        X_batch = np.array(X_batch)
        X_ids_batch = np.array(X_ids_batch)
        X_mask_batch = np.array(X_mask_batch)
        X_chars_batch = np.array(X_chars_batch)
        y_ids_batch = np.array(y_ids_batch)
        # print(X_batch[0],X_ids_batch[0],X_mask_batch[0],X_chars_batch[0],y_ids_batch)
        # print(X_chars_batch.shape)
        return X_batch, X_ids_batch,X_mask_batch,X_chars_batch,y_ids_batch

