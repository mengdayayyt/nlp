from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline

tokenizer = AutoTokenizer.from_pretrained("BubbleSheep/Hgn_trans_en2zh")

model = AutoModelForSeq2SeqLM.from_pretrained("BubbleSheep/Hgn_trans_en2zh")
translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
#translate_result = translation('I like to study Data Science and Machine Learning.', max_length=400)
#print(translate_result)
f = open('n_en.txt', 'r', encoding='utf-8')
g = open('n_result3.txt', 'w', encoding='utf-8')

line = f.readline()
while line:
    a=translation(line, max_length=400)
    b=a[0]
    g.write(b['translation_text'])
    g.write('\n')
    line = f.readline()
f.close()
g.close()    