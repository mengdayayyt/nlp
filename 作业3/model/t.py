from sacrebleu.metrics import BLEU
f = open('2000012952-岳禹彤-enzh5.txt', 'r',encoding='utf-8')
preds = []
lables = []
for line in f:
    pred, lable = line.split('\t')[1:]
    preds += [pred.strip()]
    lables += [[lable.strip()]]
blue = BLEU(tokenize= 'zh')
print(blue.corpus_score(preds, lables))