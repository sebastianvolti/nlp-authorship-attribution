from modelo_delta import distancia_manhattan,Zij,most_close
from collections import Counter
import numpy as np
import re
import pandas as pd


L=[]
L.append(open("../textos/ConAutor/L1-Conrad.txt", "r",encoding="latin_1").read())
L.append(open("../textos/ConAutor/L2-Zola.txt", "r",encoding="latin_1").read())
L.append(open("../textos/ConAutor/L3-Proust.txt", "r",encoding="latin_1").read())
L.append(open("../textos/ConAutor/L4-Austen.txt", "r",encoding="latin_1").read())
L.append(open("../textos/ConAutor/L5-Flaubert.txt", "r",encoding="latin_1").read())

T=[]
T.append(open("../textos/SinAutor/T1.txt", "r",encoding="latin_1").read())
T.append(open("../textos/SinAutor/T2.txt", "r",encoding="latin_1").read())
T.append(open("../textos/SinAutor/T3.txt", "r",encoding="latin_1").read())
T.append(open("../textos/SinAutor/T4.txt", "r",encoding="latin_1").read())
T.append(open("../textos/SinAutor/T5.txt", "r",encoding="latin_1").read())

def main():
    #data_set is the colleccion
    #i is a word (string) in the colleccion
    #j in a authorn (int) j=0 -> L1 , j=1 -> L2, j=2 -> L3, j=3 -> L4, j=4 -> L5

    data_set = L[0]+L[1]+L[2]+L[3]+L[4]
    text = re.sub('([\W_]+)|(\n)', ' ',data_set.lower())
    split_it = text.split()
    counter = Counter(split_it)
    most_occur = counter.most_common(150)
    new_list = [ seq[0] for seq in most_occur ]
    order_vector=sorted(new_list)
    #order_vector has the n word most commun

    ocurr_texts=[]
    #List that contains the vector ocurrences for each author

    for i in range(5):
        l = [0] * 150
        text = re.sub('([\W_]+)|(\n)', ' ',L[i].lower())
        split_it = text.split()
        counter = Counter(split_it)
        for i,elem in enumerate(order_vector):
            if elem in counter.keys():
                l[i]=(counter[elem])
        ocurr_texts.append(l)

    
    profile_conrad= [0] * 150
    profile_zola= [0] * 150
    profile_proust= [0] * 150
    profile_austen= [0] * 150
    profile_flaubert= [0] * 150

    #Profiling authors
    for i in range(150):
        profile_conrad[i] = Zij(0,i,ocurr_texts)
        profile_zola[i] = Zij(1,i,ocurr_texts)
        profile_proust[i] = Zij(2,i,ocurr_texts)
        profile_austen[i] = Zij(3,i,ocurr_texts)
        profile_flaubert[i] = Zij(4,i,ocurr_texts)

    ocurr_texts_without_author=[]
    #List that contains the vector ocurrences for each instance without author

    for i in range(5):
        t = [0] * 150
        text = re.sub('([\W_]+)|(\n)', ' ',T[i].lower())
        split_it = text.split()
        counter = Counter(split_it)
        for i,elem in enumerate(order_vector):
            if elem in counter.keys():
                t[i]=(counter[elem])
        ocurr_texts_without_author.append(t)

    #Calculate best author
    profile_t1= [0] * 150
    profile_t2= [0] * 150
    profile_t3= [0] * 150
    profile_t4= [0] * 150
    profile_t5= [0] * 150
        
    #Profiling instance without author
    for i in range(150):
        profile_t1[i] = Zij(0,i,ocurr_texts_without_author)
        profile_t2[i] = Zij(1,i,ocurr_texts_without_author)
        profile_t3[i] = Zij(2,i,ocurr_texts_without_author)
        profile_t4[i] = Zij(3,i,ocurr_texts_without_author)
        profile_t5[i] = Zij(4,i,ocurr_texts_without_author)
    
    #Evaluation
    distances_conrad=[distancia_manhattan(profile_conrad,profile_t1),distancia_manhattan(profile_conrad,profile_t2),distancia_manhattan(profile_conrad,profile_t3),distancia_manhattan(profile_conrad,profile_t4),distancia_manhattan(profile_conrad,profile_t5)]
    distances_zola=[distancia_manhattan(profile_zola,profile_t1),distancia_manhattan(profile_zola,profile_t2),distancia_manhattan(profile_zola,profile_t3),distancia_manhattan(profile_zola,profile_t4),distancia_manhattan(profile_zola,profile_t5)]
    distances_proust=[distancia_manhattan(profile_proust,profile_t1),distancia_manhattan(profile_proust,profile_t2),distancia_manhattan(profile_proust,profile_t3),distancia_manhattan(profile_proust,profile_t4),distancia_manhattan(profile_proust,profile_t5)]
    distances_austen=[distancia_manhattan(profile_austen,profile_t1),distancia_manhattan(profile_austen,profile_t2),distancia_manhattan(profile_austen,profile_t3),distancia_manhattan(profile_austen,profile_t4),distancia_manhattan(profile_austen,profile_t5)]
    distances_flauber=[distancia_manhattan(profile_flaubert,profile_t1),distancia_manhattan(profile_flaubert,profile_t2),distancia_manhattan(profile_flaubert,profile_t3),distancia_manhattan(profile_flaubert,profile_t4),distancia_manhattan(profile_flaubert,profile_t5)]

    a = np.array([distances_conrad,distances_zola,distances_proust,distances_austen,distances_flauber])
    df=pd.DataFrame(a,columns=['conrad','zola','proust','austen','flauber'],index=['T1','T2','T3','T4','T5'])
    print(df)

    print('\n')
    print('For the instance t1,most close:')
    print(most_close(profile_conrad,profile_zola,profile_proust,profile_austen,profile_flaubert,profile_t1))

    print('For the instance t2,most close:')
    print(most_close(profile_conrad,profile_zola,profile_proust,profile_austen,profile_flaubert,profile_t2))

    print('For the instance t3,most close:')
    print(most_close(profile_conrad,profile_zola,profile_proust,profile_austen,profile_flaubert,profile_t3))

    print('For the instance t4,most close:')
    print(most_close(profile_conrad,profile_zola,profile_proust,profile_austen,profile_flaubert,profile_t4))

    print('For the instance t5,most close:')
    print(most_close(profile_conrad,profile_zola,profile_proust,profile_austen,profile_flaubert,profile_t5))


if __name__ == '__main__':
    main()
