import os
import shutil
import docxpy
import numpy as np
import pandas as pd

folders = os.listdir()
filepaths = []
for folder in folders :
    if folder.startswith('.') or folder.endswith('.py'):
        continue
    subdirs = os.listdir(folder)
    for subdir in subdirs:
        if subdir.startswith('.'):
            continue
        for filepath in os.listdir(os.path.join(folder,subdir)):
            if not filepath.endswith('.docx'):
            #if filepath.startswith('.') or filepath.endswith('.png') or filepath.endswith('g'):
                continue
            filepaths.append(os.path.join(folder,subdir,filepath))

if not os.path.isdir('extracted'):
    os.mkdir('extracted')


texts =  []
for i,path in enumerate(filepaths):
    text = docxpy.process(path)
    if len(text)!=0:
        texts.append(text)

df = pd.DataFrame(texts,columns = ['document'])
df.to_csv('./extracted/extectedText.csv')


texts = [t.replace('\n','').replace(' ','') for t in texts if len(t)>0]
count = 0
with open('output.txt','w',encoding='utf-8') as f:
    for line in texts:
        count +=len(line)
        f.write(line)

print('write %d words'%count)
