# -*- coding: utf-8 -*-
from ckiptagger import data_utils
from ckiptagger import WS, POS, NER
#data_utils.download_data_gdown("./")

with open('output.txt','r') as f:
    text = f.read()

#text = '傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'
ws = WS("./data")
pos = POS("./data")
ner = NER("./data")

ws_results = ws([text])
pos_results = pos(ws_results)
ner_results = ner(ws_results, pos_results)
filtertext = []
#for name in ner_results[0]:
#    if name[-2] in filterlist:
#        filtertext.append(name[-1])
#        

result = []
for n in ner_results[0]:
    id1,id2,pos_,word = list(n)
    result.append(pos_+','+word)
result = '\n'.join(result)
with open("ner_output.txt",'w') as f:
    f.write(result)

print('Done')
