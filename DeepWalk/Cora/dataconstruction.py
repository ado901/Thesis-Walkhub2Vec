from turtle import title
import pandas as pd
import re
class Paper:
    def __init__(self, link, id,year,targets, label) -> None:
        self.id= id
        self.link= link
        self.year=year
        self.label=label
        self.targets= targets
        #self.month=month
papers=[]
f2=open('data/classifications')

""" with open('data/papers') as f1:
    lines= f1.readlines()
    for idx,line in enumerate(lines):
        try:
            print(idx)
            year = re.compile('<year>(.*?)</year>').search(line).group(1).strip()
            year=re.sub("[^0-9]", "", year)
            if not year.isdigit() or len(year)!=4:
                raise Exception(year+ " is not year")

            id=re.search(r'\d+', line).group()
            if any(x.id == id for x in papers):
                raise Exception(id+' already present')
            
            
            # Extracting the link from the line.
            link=re.search(r'\t(.*?)\t', line).group().strip()
            found=False
            label=None
            f2.seek(0)    
            for lineclass in f2.readlines():
                if link in lineclass:
                    label=re.search('/(.*?)/',lineclass).group()
                    label =label.replace("/","")

                    found=True
            if not found:
                raise Exception(link + " no occurrence in classifications")
            targets=[]
            
            paper= Paper(link,id, year, targets,label)
           
            papers.append(paper)
            print("ok ",paper.id,paper.link,paper.year, label)
            
        except Exception as e:
            print(e)
            continue
f2.close()
import csv
with open('nodescomplete.csv','w', newline='') as csvfile:
    writer= csv.writer(csvfile)
    writer.writerow(['id','Label','Year'])
    for paper in papers:
        writer.writerow([paper.id,paper.label,paper.year])
import csv
f3=open('data/citations')
fnodes=open('nodescomplete.csv')

papers=[]
spamreader = csv.reader(fnodes, delimiter=',')
next(spamreader, None)  # skip the headers
for row in spamreader:
    papers.append(Paper('',row[0],row[2],[],row[1]))
fnodes.close()
alreadysawid=[]
currentid= None
previouspaper=None
for n,paper in enumerate(papers):
    print(n/len(papers)*100)
    f3.seek(0)
    for line in f3:
        line=line.strip()
        line=line.split('\t')
        if line[0] ==paper.id and any(x.id==line[1] for x in papers):
            paper.targets.append(line[1])

f3.close()

with open('nodes.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','Label'])
    for paper in papers:
        writer.writerow([paper.id,paper.label])
with open('edges.csv','w',newline='') as csvfile:
    writer= csv.writer(csvfile)
    writer.writerow(['Source','Target','Year'])
    for paper in papers:
        for target in paper.targets:
            writer.writerow([paper.id,target,paper.year])
import numpy as np
edges=pd.read_csv('edges.csv')
print(len(pd.unique(edges['Source'])))
nodes=pd.read_csv('nodes.csv')
print(len(pd.unique(nodes['id'])))
count=0
Col1_value = set(edges['Target'].unique())
#print(np.logical_not(nodes['id'].isin(edges['Source'])))
nodes=nodes[np.logical_or((nodes['id'].isin(edges['Source'])), (nodes['id'].isin(edges['Target'])))]
with open("nodes.csv","w+", newline='') as f1:
    nodes.to_csv(f1, index=False)
for n,i in enumerate(nodes['id']):
    #print(n/len(edges['Source'])*100)
    if i not in Col1_value:
        print(i)
        count+=1
print(count)
 """

#-------------------------
nodescomplete= pd.read_csv('nodescomplete.csv')
nodes= pd.read_csv('nodes.csv')
nodescomplete=nodescomplete[nodescomplete['id'].isin(nodes['id'])]
with open("nodescomplete.csv","w+", newline='') as f1:
    nodescomplete.to_csv(f1, index=False)

