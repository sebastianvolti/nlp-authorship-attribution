import statistics
import re
from collections import Counter


def Fij(j,i,ocurr_texts):    
    return ocurr_texts[j][i]

def Dj(j,ocurr_texts):
    sum=0
    for i in range(150):
        sum+=Fij(j,i,ocurr_texts)
    return sum
    

def FRij(j,i,ocurr_texts):
    return Fij(j,i,ocurr_texts)/Dj(j,ocurr_texts)


def media(i,ocurr_texts):
    FR=[]
    for j in range(5):
        FR.append(FRij(j,i,ocurr_texts))
    return statistics.mean(FR)
    
    

def desviacioni(i,ocurr_texts):
    FR=[]
    for j in range(5):
        FR.append(FRij(j,i,ocurr_texts))
    return statistics.stdev(FR)


def Zij(j,i,ocurr_texts):
    if desviacioni(i,ocurr_texts)==0: return 0
    else: return (FRij(j,i,ocurr_texts)-media(i,ocurr_texts))/desviacioni(i,ocurr_texts)


def distancia_manhattan(x,y):
    sum=0
    for i in range(150):
        dist=abs(x[i]-y[i])
        sum+=dist
    return sum

def most_close(conrad,zola,proust,austen,flaubert,x):
    dist=[]
    dist.append(distancia_manhattan(conrad,x))
    dist.append(distancia_manhattan(zola,x))
    dist.append(distancia_manhattan(proust,x))
    dist.append(distancia_manhattan(austen,x))
    dist.append(distancia_manhattan(flaubert,x))
    if dist.index(min(dist)) == 0:
        return 'Conrad'
    elif dist.index(min(dist)) == 1:
        return 'Zola'
    elif dist.index(min(dist)) == 2:
        return 'Proust'
    elif dist.index(min(dist)) == 3:
        return 'Austen'
    elif dist.index(min(dist)) == 4:
        return 'Flauber'
    else:
        return 'none'
