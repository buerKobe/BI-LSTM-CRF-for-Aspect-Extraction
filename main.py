import tensorflow as tf
from model import BiLSTM_CRF
import numpy as np
import os,argparse,time,random
import pandas as pd
import re
import eval_test
import matplotlib.pyplot as plt
from pre_data import read_corpus,read_dictionary,tag2label,random_embedding,str2bool,get_logger

#hyperparmeters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for aspect extraction')
parser.add_argument('--train_data',type=str,default='data',help='train data source')
parser.add_argument('--dev_data',type=str,default='data',help='dev data source')
parser.add_argument('--test_data',type=str,default='data',help='test data source')
parser.add_argument('--batch_size',type=int,default=64,help='#sample of each minibatch')
parser.add_argument('--epoch',type=int,default=64,help='#epoch of training')
parser.add_argument('--hidden_dim',type=int,default=300,help='#dim of hidden state')
parser.add_argument('--optimizer',type=str,default='Adam',help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF',type=str2bool,default=True,help='use CRF at the top layer.if False use softmax')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--clip',type=float,default=5.0,help='gradient clipping')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout keep_prob')
parser.add_argument('--update_embedding',type=str2bool,default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding',type=str,default='random',help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim',type=int,default=300,help='random init char embedding_dim')
parser.add_argument('--shuffle',type=str2bool,default=True,help='shuffle training data before each epoch')
parser.add_argument('--mode',type=str,default='train',help='train/test/demo')
parser.add_argument('--flag',type=int,default=0,help='flag of BiLSTM/0 or LSTM/1')
args = parser.parse_args()
args.flag = 0
args.CRF = True
#args.hidden_dim = 100
args.batch_size = 64
#args.lr = 0.001
#args.epoch = 100
#args.mode = 'test'

word2id = read_dictionary(os.path.join('.',args.train_data,'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id,args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path),dtype='float32')

if args.mode != 'test':
    train_path = os.path.join('.',args.train_data,'trains.csv')
    train_data = read_corpus(train_path)
    dev_path = os.path.join('.', args.dev_data, 'devs.csv')
    dev_data = read_corpus(dev_path)
    dev_size = len(dev_data)
    train_size = len(train_data)


#path setting
timestamp = str(int(time.time())) if args.mode == 'train' else 'result'
output_path = os.path.join('.',args.train_data+'_save',timestamp + '_LSTM')
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path,'summaries')
if not os.path.exists(summary_path):os.makedirs(summary_path)
model_path = os.path.join(output_path,'checkpoints')
if not os.path.exists(model_path):os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path,'model')
result_path = os.path.join(output_path,'results')
if not os.path.exists(result_path):os.makedirs(result_path)
log_path = os.path.join(result_path,'log.txt')
get_logger(log_path).info(str(args))

if args.mode == 'train':
    print(ckpt_prefix)
    model = BiLSTM_CRF(batch_size=args.batch_size,epoch_num=args.epoch,hidden_dim=args.hidden_dim,embeddings=embeddings,
                       dropout_keep=args.dropout,optimizer=args.optimizer,lr=args.lr,clip_grad=args.clip,
                       tag2label=tag2label,vocab=word2id,shuffle=args.shuffle,model_path=ckpt_prefix,
                       summary_path=summary_path,log_path=log_path,result_path=result_path,flag=args.flag,CRF=args.CRF,update_embedding=args.update_embedding)
    model.build_graph()

    print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    print('train data: {}'.format(len(train_data)))
    model.train(train=train_data, dev=dev_data)

else:
    test_path = os.path.join('.', args.test_data, 'tests.csv')
    test_data = read_corpus(test_path)
    test_size = len(test_data)
    print("test data: {}".format(test_size))
    model_path = './data_save/1531191236_LSTM/checkpoints'
    ckpt_prefix = tf.train.latest_checkpoint(model_path)
    model = BiLSTM_CRF(batch_size=args.batch_size,epoch_num=args.epoch,hidden_dim=args.hidden_dim,embeddings=embeddings,
                       dropout_keep=args.dropout,optimizer=args.optimizer,lr=args.lr,clip_grad=args.clip,
                       tag2label=tag2label,vocab=word2id,shuffle=args.shuffle,model_path=ckpt_prefix,
                       summary_path=summary_path,log_path=log_path,result_path=result_path,flag=args.flag,CRF=args.CRF,update_embedding=args.update_embedding)
    model.build_graph()
    print(len(dev_data))
    model.test(test_data)

    '''
    # generate result.txt
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('============= test =============')
        saver.restore(sess, ckpt_prefix)
        with open(test_path,encoding='utf-8') as fr:
            lines = fr.readlines()
            # result = {}
            # result['char'] = []
            # result['label'] = []
            # result['tag'] = []
            words = []
            labels = []
            file = open('result.txt', 'w', encoding='utf-8')
            for line in lines:
                line = line.strip().strip('\n')
                word,label = line.split(',')
                if word != '':
                    words.append(word)
                    labels.append(label)
                else:
                    demo_data = [(words,['O'] * len(words))]
                    tag = model.demo_one(sess,demo_data)
                    for i, (char, label_, tag_) in enumerate(zip(words, labels, tag)):
                        file.write(char+'\t'+label_+'\t'+tag_)
                        file.write('\n')
                    file.write('\n')
                    words = []
                    labels = []
                # result.csv
            #         for i,(char,label_,tag_) in enumerate(zip(words,labels,tag)):
            #             result['char'].append(char)
            #             result['label'].append(label_)
            #             result['tag'].append(tag_)
            #         result['char'].append('')
            #         result['label'].append('')
            #         result['tag'].append('')
            #         words = []
            #         labels = []
            # print(len(words))
            # result = pd.DataFrame(result)
            # # print(result)
            # result.to_csv('result.csv',index=False, header=None)
            '''
