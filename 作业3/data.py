
en=open('news_commentary.txt','r',encoding='utf-8')
g = open('n_en.txt', 'w', encoding='utf-8')
f = open('n_zh.txt', 'w', encoding='utf-8')
text =en.read()
a=eval(text)
print(','.join(map(str, sorted(a.keys()))))
lb=a['rows']
for zd in lb:
    zdd=zd['row']
    zddd=zdd['translation']
    a=zddd['en']
    b=zddd['zh']
    g.write(a)
    g.write('\n')
    f.write(b)
    f.write('\n')
en.close()
g.close()
f.close()



