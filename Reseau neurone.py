"""NOTE: Le rajout du coeff de biais a considérablement amélioré le réseau! Pour la fction XOR, l'EQM mini obtenu après 1000 ité est 0.05, avec le coeff debiais on est à 0.0001 en 6 ité!!!"""

# 0 <= nochouche <= len(nbneurone) , et nocouche=0 correspond à l'entrée, nocouche = 1 correspond à la première couche de neurone
#poids = [liaison nocouche -> nochouche+1][noneurone de la couche nochouche+1][noentre]
#nbneurone[nocouche] = nbneurone de la couche nocouche, nbneurone[0] correspond donc au nombre d'entrées du réseau!!
#sortie = [nocouche][noneurone] correspond à la sortie du neurone. Pour, sortie[0], on parle donc des valeurs en entrée du réseau
nbneurone = [1,10]

from random import *
from math import *
import matplotlib.pyplot as pt
import copy

"""Idee: liste qui contient les sorties de chaque neurones et dont la première case contient l'entrée du réseau.
Il y aura donc une fonction de propagation qui aura pour but de caluler les sorties de la couche suivante de neurones"""

def genpoidsalea(nbneurone):
    poids = []
    for nocouche in range(0,len(nbneurone)-1):
        poids.append([])
        for noneurone in range(0,nbneurone[nocouche+1]):
            poids[nocouche].append([])
            for noentre in range(0,nbneurone[nocouche]+1): #le +1 est du aux coefficient de biais: s = sum(ei*wi)+coeffbiais
                poids[nocouche][noneurone].append((random()-0.5)*2) #on fixe le coeef de biais: c'est le dernier coeff. les autres coeff ont ds la case d'indice associée à l'entree
    return poids
    
def gensortie(nbneurone):
    sortie = []
    for i in nbneurone:
        sortie.append([0]*i)
    return sortie

def sigm(x):
    if x<-500:
        return 0
    elif x>500:
        return 1
    else:
        return 1/(1+exp(-x))

def neurone(poids, sortie, nocouche, noneurone): #retourne la sortie d'un neurone
    entre = sortie[nocouche-1]
    pds = poids[nocouche-1][noneurone]
    if len(entre) != len(pds)-1: #pendant les tests!!
        raise IndexError
    sum = 0
    for i in range(len(entre)):
        sum += entre[i]*pds[i]
    sum += pds[-1] #on rajoute le coeff de biais
    return sigm(sum)
    
def propagation(poids, sortie, nocouche, nbneurone): #retourne la liste sortie completée, calcule la sortie de la couche nocouche en fonction de celle de nocouche-1
    for noneurone in range(nbneurone[nocouche]):
        sortie[nocouche][noneurone] = neurone(poids, sortie, nocouche, noneurone)
    return sortie

def reseau(poids, sortie, nbneurone): #seule sortie[0] est remplie, le reste de la liste est obsolète. Reseau la revoie completée
    for nocouche in range(1,len(nbneurone)):
        sortie = propagation(poids, sortie, nocouche, nbneurone)
    return sortie

##Algo génétique d'apprentissage: population de 40 réseaux. La génération suivante comprend le meilleur des réseaux et ce même réseau muté

def mute(poids,amplitude=30,probmutation=0.8): #promutation : probabilité qu'un poids mute; amplitude: valeur max de fluctuation du poids
    lst = copy.deepcopy(poids) #deepcopy obligé pour eviter les alias avec poids...
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            for k in range(len(lst[i][j])):
                if random()<probmutation:
                    lst[i][j][k] += amplitude*2*(random()-0.5)
    return lst
    
def gengeneration(meilleur,population): #meilleur=meilleur poids de la génération précedente
    generation = [meilleur]
    for i in range(1,population):
        nouveau = mute(meilleur)
        generation.append(nouveau)
    return generation
    
def EQ(sortie): #determine l'erreur quadratique du réseau
    resreseau = sortie[-1]
    resvoulu = loi(sortie[0])
    if len(resreseau) != len(resvoulu): # juste pr les test!!
        raise IndexError
    sum = 0
    for i in range(len(resreseau)):
        sum += (resreseau[i]-resvoulu[i])**2
    return sum

def EQM(poids,nbneurone,nbtest = 30): #détermine le carré de l'erreur quadratique moyenne d'un réseau; nbtest = nombre de sortie de réseau calculée
    sum = 0
    for i in range(nbtest):
        sortie = gensortie(nbneurone)
        sortie[0] = test()
        sortie = reseau(poids, sortie, nbneurone)
        sum += EQ(sortie)
    return sum/nbtest
    
def apprentissage(nbneurone, population = 100, EQMvoulu = 0.01, generationmax = 100): #generationmax = val de la gene max pr eviter boucle infinie si jamais EQM non atteint
    generation = [genpoidsalea(nbneurone) for i in range(population)]
    #On détermine quel poids donne le meilleur EQM
    nobestEQM = 0 #membre de la pop avec le meilleur EQ
    bestEQM = EQMvoulu + 1
    nogeneration = 0
    X = []
    Y = []
    while bestEQM>EQMvoulu and nogeneration<=generationmax:
        nobestEQM= 0
        bestEQM = EQM(generation[0],nbneurone)
        for nomembre in range(1,population):
            EQMmembre = EQM(generation[nomembre],nbneurone)
            if EQMmembre < bestEQM:
                bestEQM = EQMmembre
                nobestEQM = nomembre
        generation = gengeneration(generation[nobestEQM],population)
        if nogeneration%10 == 0:
            print("Generation ",nogeneration)
        X.append(nogeneration)
        Y.append(bestEQM)
        nogeneration += 1
    pt.plot(X,Y)
    pt.show()
    return (generation[0], nogeneration, bestEQM)
        
## Définition de la LOI du réseau (sortie en fonction de l'entrée) et de la base de TEST. Ici la loi est: le réseau active la sortie dont le numéro est celui de la 1° decimale de l'entree

def loi(entre): #renvoie ce que l'on veut dans sortie[-1], en foncton de ce que l'on a en entre = sortie[0]  (on renvoie donc une liste!! et on donne en entree une liste!!)
    a = int(entre[0]*10)
    res = [0]*10
    res[a] = 1
    return res

def test(): #renvoie ce que l'on doit mettre dans sortie[0]
    return [random()]
    
"""def loi(entre):
    [a,b] = entre
    return [int((a*b==0 and a+b==1))]

vartest = 0

def test(): #avec vartest, chaque appel de test() revoie la donnée suivante
    global vartest
    vartest += 1
    list = ([0,0],[1,1],[0,1],[1,0])
    return list[vartest%4]"""
    
##
def ex(poids,nbneurone,entre):
    sortie = gensortie(nbneurone)
    sortie[0] = [entre]
    sortie = reseau(poids,sortie,nbneurone)
    m = sortie[-1]
    return sortie