# -*- coding:utf-8 -*-
import os
import time
import operator
import json
import numpy as np
weight={}

def get_feature(words):
    features=[]
    for i,word in enumerate(words):
        left2=words[i-2] if i>1 else '*'
        left1=words[i-1] if i>0 else '*'
        mid = word
        right1=words[i+1] if i+1<len(words) else '*'
        right2=words[i+2] if i+2<len(words) else '*'
        feature=['1_'+left1,'2_'+mid,'3_'+right1,'4_'+left2+'+'+left1,
        '5_'+left1+'+'+mid,'6_'+mid+'+'+right1,'7_'+right1+'+'+right2]
        features.append(feature)
    return features
def get_weight(key):
    if weight.get(key)==None:
        weight[key]=0
    return weight[key]
def update_weight(key,val):
    if weight.get(key)==None:
        weight[key]=0
    weight[key]+=val
def update_weights(x,gt_y,pred_y):
    features=get_feature(x)
    for i in range(len(gt_y)-1):
        update_weight(str(gt_y[i])+'_'+str(gt_y[i+1]),1)
    for i in range(len(x)):
            for feature in features[i]:
                update_weight(str(gt_y[i])+'_'+feature,1)
    for i in range(len(pred_y)-1):
        update_weight(str(pred_y[i])+'_'+str(pred_y[i+1]),-1)
    for i in range(len(x)):
            for feature in features[i]:
                update_weight(str(pred_y[i])+'_'+feature,-1)

def read_sentence(sentence): #bmes
    y=[]
    for word in sentence:
        if len(word) == 1:
            y.append(3)
        else:
            y+=([0]+[1]*(len(word)-2)+[2])
    x = ''.join(sentence)
    return x,y
def get_sentence(x,y):
    words=[]
    word=''
    for i,w in enumerate(x):
        word+=w
        if y[i]==2 or y[i]==3:
            words.append(word)
            word=''
    if word != '':
        words.append(word)
    return words
def decode(x):
    #print(x)
    transition=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            transition[i][j]=get_weight(str(i)+'_'+str(j))
    emission=np.zeros((len(x),4))
    features=get_feature(x)
    for i in range(len(x)):
        for j in range(4):
            total=0
            for feature in features[i]:
                total+=get_weight(str(j)+'_'+feature)
            emission[i][j]=total
    #print(len(x))
    viterbi=np.zeros((len(x),4))
    record=np.zeros((len(x),4))
    viterbi[0] = emission[0]
    for i in range(1,len(x)):
        for j in range(4):
            # k状态转移到j状态
            max_score=-10000000
            max_k=0
            for k in range(4):
                score = viterbi[i-1][k] + transition[k][j] + emission[i][j]
                if score > max_score:
                    max_score=score
                    max_k=k
            viterbi[i][j]=max_score
            record[i][j]=max_k 
    #print(viterbi)
    #print(record)
    best_path=np.zeros(len(x),dtype=int)
    best_path[-1]=np.argmax(viterbi[len(x)-1])
    for i in range(len(x)-1,0,-1):
        best_path[i-1]=record[i][best_path[i]]
    #print(list(best_path))
    return list(best_path)

def evaluate(file):
    with open(file,'r',encoding='utf-8') as f:
        TP,TPFP,TPFN=0,0,0
        delta=0.000001
        for line in f.readlines():
            words = line.rstrip().split('  ')
            x=''.join(words)
            pred_y=decode(x)
            #print(pred_y)
            pred=get_sentence(x,pred_y)
            #print(' '.join(words))
            #print(' '.join(pred))
            now=0
            for i,word in enumerate(words):
                words[i]=str(now)+'_'+words[i]
                now+=len(word)
            now=0
            for i,word in enumerate(pred):
                pred[i]=str(now)+'_'+pred[i]
                now+=len(word)
            TPFP+=len(pred)
            TPFN+=len(words)
            TP+=len(set(words).intersection(set(pred)))
        recall = TP/(TPFN+delta)
        precision = TP/(TPFP+delta)
        return recall,precision
def train():
    epochs=10
    with open('train.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    for epoch in range(epochs):
        st = time.time()
        for i,line in enumerate(lines):
            words = line.rstrip().split('  ')
            x,gt_y=read_sentence(words)
            if len(x) <= 0:
                continue
            pred_y=decode(x)
            if pred_y!=gt_y:
                update_weights(x,gt_y,pred_y)
        if epoch == 0:
            print('total features = {}'.format(len(weight.keys())))
        ed = time.time()
        with open('backup/epoch_{}.json'.format(epoch), 'w',encoding='utf-8') as json_file:
            json.dump(weight,json_file,ensure_ascii=False)
        recall,precision = evaluate('dev.txt')
        print('Epoch[{}] Time:{:.3f}s Recall:{:.3f} Precision:{:.3f} F1:{:.3f}'.format(epoch,ed-st,recall,precision,2*recall*precision/(recall+precision)))
def test():
    global weight
    with open('backup/epoch_6.json','r',encoding='utf-8') as json_file:
        weight = json.load(json_file)
    with open('result.txt','w',encoding='utf-8') as dst:
        with open('dev.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                x=''.join(line.rstrip().split('  '))
                pred_y=decode(x)
                pred=get_sentence(x,pred_y)
                #print(pred)
                dst.write('  '.join(pred)+'\n')

#test()
train()