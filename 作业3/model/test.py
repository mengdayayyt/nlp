# prefect match
from nltk.translate.bleu_score import sentence_bleu


f = open('n_zh.txt', 'r', encoding='utf-8')
g = open('n_result.txt', 'r', encoding='utf-8')

s=0
i=0
line1 = f.readline()
line2 = g.readline()
while line1:
    reference = []
    for x in range(0, len(line1)):
        reference.append(line1[x])
    reference1 = []
    reference1.append(reference)
    candidate = []
    for x in range(0, len(line2)):
        candidate.append(line2[x])
    score = sentence_bleu(reference1, candidate)
    print(score)
    s=s+score
    i=i+1
    line1 = f.readline()
    line2 = g.readline()
print(s/i)
f.close()
g.close()
