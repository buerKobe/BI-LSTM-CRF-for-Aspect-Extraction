import sys,pickle,os,random
import logging, sys, argparse
import numpy as np
import codecs
import pandas as pd
import csv



tag2label = {
    'O':0,
    # "B-PER.NAM": 1, "I-PER.NAM": 2,
    # "B-PER.NOM": 3, "I-PER.NOM": 4,
    # "B-LOC.NAM": 5, "I-LOC.NAM": 6,
    # "B-LOC.NOM": 5, "I-LOC.NOM": 6,
    # "B-ORG.NAM": 9, "I-ORG.NAM": 10,
    # "B-GPE.NAM": 11, "I-GPE.NAM": 12
    'B-ASP':1,
    'I-ASP':2

}



def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return:
    """
    data = []
    with codecs.open(corpus_path,encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_,tag_ = [],[]
    for line in lines:
        if line != ',\r\n':
            [char,label] = line.strip().split(',')
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_,tag_))
            sent_,tag_ = [],[]
    return data

def vocab_build(vocab_path,corpus_path,min_count):
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_,tag_ in data[1:]:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1,1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word,[word_id,word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['PAD'] = 0

    with open(vocab_path,'wb') as fw:
        pickle.dump(word2id,fw)

def sentence2id(sent,word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path,'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:',len(word2id))
    return word2id

def random_embedding(vocab,embedding_dim):
    embedding_mat = np.random.uniform(-0.25,0.25,(len(vocab),embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def pad_sequences(sequences,pad_mark=0):
    max_len = max(map(lambda x:len(x),sequences))
    seq_list,seq_len_list = [],[]
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq),0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq),max_len))
    return seq_list,seq_len_list


def batch_yield(data,batch_size,vocab,tag2label,shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs,labels = [],[]
    for (sent_,tag_) in data:
        try:
            sent_ = sentence2id(sent_,vocab)
            label_ = [tag2label[tag] for tag in tag_]
        except KeyError:
            continue
        if len(seqs) == batch_size:
            yield seqs,labels
            seqs,labels = [],[]

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0 :
        yield seqs,labels

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # vocab_build('./data/word2id.pkl','./data/train.csv',1)
    data = read_corpus('./data/tests.csv')
    print(len(data),data)
    #word2id = read_dictionary('./data_path/word2id.pkl')
    #batch_yield(data,64,word2id,tag2label,shuffle=True)
    #print(data)

