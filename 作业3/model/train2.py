from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,pipeline

mode_name = 'trans-opus-mt-en-zh'
model = AutoModelForSeq2SeqLM.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
#translate_result = translation('I like to study Data Science and Machine Learning.', max_length=400)
#print(translate_result)
f = open('o_en.txt', 'r', encoding='utf-8')
g = open('o_result2.txt', 'w', encoding='utf-8')

translator = pipeline("translation_en_to_zh",model='Helsinki-NLP/opus-mt-en-zh')
line = f.readline()
while line:
    a=translation(line, max_length=400)
    b=a[0]
    g.write(b['translation_text'])
    g.write('\n')
    line = f.readline()
f.close()
g.close()    