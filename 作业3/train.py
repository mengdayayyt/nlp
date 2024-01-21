from transformers import pipeline

f = open('n_en.txt', 'r', encoding='utf-8')
g = open('n_result2.txt', 'w', encoding='utf-8')

translator = pipeline("translation_en_to_zh",model='Helsinki-NLP/opus-mt-en-zh')
line = f.readline()
while line:
    a=translator(line)
    b=a[0]
    g.write(b['translation_text'])
    g.write('\n')
    line = f.readline()
    print(1)
f.close()
g.close()
# a=translator("How old are you?")

