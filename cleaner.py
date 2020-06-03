import re
pattern_sid = r'[ABMabm]\d{7}\d+'
pattern_class = r'[四二]\S{2}[一二三四甲乙丙]'
pattern_class1 =  r'[四二]\S{}[一二三四甲乙丙]'
pattern_continuespace = r'\s+'
pattern_symbol = r'[!@/#$%^&*()_+=\"\'<>?;:\{\}『…』「」“〃,`~\\|,（）：–.]'
pattern_eng = r'[a-z]+'
pattern_day = r'\d+/\d+'
pattern_other = r'企管\S*'


ner_filter = ['CARDINAL','PERSON','ORG']
#'[+_)(*&^%$#@!~`=)]'


with open('output.txt','r') as f:
    text = f.read().lower()
with open('ner_output.txt','r') as f:
    ner = f.read()
ner = ner.split('\n')
ner = [line.split(',') for line in ner]
ner = [noun for pos,noun in ner if pos in ner_filter]
for n in set(ner):
    text = text.replace(n,'')

#
#sid  = re.findall(pattern_sid, text)
#classname = re.findall(pattern_class,text)
#symbol = re.findall(pattern_symbol, text)
#eng = re.findall(pattern_eng, text)
#
text = re.sub(pattern_sid , ''  ,text)
text = re.sub(pattern_class, "", text)
text = re.sub(pattern_continuespace," ",text)
text = re.sub(pattern_symbol, "", text)
text = re.sub(pattern_eng, "" ,text)
text = re.sub(pattern_day, "", text)
text = re.sub(pattern_other, "" ,text)

text = text.replace('2018','')
text = text.replace('行銷資料科學',"")
text = text.replace('孟彥','')
text = text.replace('學號','')
text = text.replace('黃丹禕','')
text = text.replace('廖靜芸','')
text = text.replace('潔思','')
text = text.replace('方聖瑋','')
text = text.replace('0','')

with open('mds.txt','w') as f:
    f.write(text)
print('done')
