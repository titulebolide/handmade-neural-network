"""Idée: on entre une image en entrée en noir et blanc (en pratique on garde la nuance rouge), et le reseau active la sortie qui est associée à l'attribut de l'image, inhibe les autres. La sortie i du reseau a comme attribut attribut[i]"""

##IMPORTS
import os
from random import *
from math import *
import matplotlib.pyplot as pt
import copy
import imageio as im

##REPERTOIRE DE LA BASE D'ENTRAINEMENT
direc = """/home/titus/Programmation/Python/Réseau de neurone algo génétique/Reconaissance d'image/data/"""  #localistation du dossier data, dans lequel les fichiers d'entraïnement sont classés dans des dossiers portant le nom de leurs attribut

##CONSTRUCTION DE LA BASE D'APPRENTISSAGE
error = False
attribut = []
try :
    attribut = os.listdir(direc)
except FileNotFoundError:
    error = True
if attribut == []:    error = True
if error == True :  print("""Dossier "data" invalide ou incomplet""")

##VARIABLES GLOBALES
pixh = 30 #hauteur des images considérées en pixels rogne ou remplit le reste par du noir
pixl = 20 #largeur
nbneurone = [pixh*pixl,50,len(attribut)] #strucure du reseau de neurone : nb entrees, nb neurones couches 1, 2 ... n et couche n = sortie
population = 10 #nb reseau par generations
nbtest = 10 #nb de photos testées par attributs pr determiner fitness

## RESEAU DE NEURONE

# 0 <= nochouche <= len(nbneurone) , et nocouche=0 correspond à l'entrée, nocouche = 1 correspond à la première couche de neurone
#poids = [liaison nocouche -> nochouche+1][noneurone de la couche nochouche+1][noentre]
#nbneurone[nocouche] = nbneurone de la couche nocouche, nbneurone[0] correspond donc au nombre d'entrées du réseau!!
#sortie = [nocouche][noneurone] correspond à la sortie du neurone. Pour, sortie[0], on parle donc des valeurs en entrée du réseau

def genpoidsalea():
    poids = []
    for nocouche in range(0,len(nbneurone)-1):
        poids.append([])
        for noneurone in range(0,nbneurone[nocouche+1]):
            poids[nocouche].append([])
            for noentre in range(0,nbneurone[nocouche]+1): #le +1 est du aux coefficient de biais: s = somme(ei*wi)+coeffbiais
                poids[nocouche][noneurone].append((random()-0.5)*2) #on fixe le coeef de biais: c'est le dernier coeff. les autres coeff ont ds la case d'indice associée à l'entree
    return poids
    
def gensortie():
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
    som = 0
    for i in range(len(entre)):
        som += entre[i]*pds[i]
    som += pds[-1] #on rajoute le coeff de biais
    return sigm(som)
    
def propagation(poids, sortie, nocouche): #retourne la liste sortie completée, calcule la sortie de la couche nocouche en fonction de celle de nocouche-1
    for noneurone in range(nbneurone[nocouche]):
        sortie[nocouche][noneurone] = neurone(poids, sortie, nocouche, noneurone)
    return sortie

def reseau(poids, sortie): #seule sortie[0] est remplie, le reste de la liste est obsolète. Reseau la revoie completée
    for nocouche in range(1,len(nbneurone)):
        sortie = propagation(poids, sortie, nocouche)
    return sortie
    
def test(poids,nomim,attri): #attribut, numero image, retourne la sortie de réseau
    sortie = gensortie()
    image = im.imread(direc+attri+"/"+nomim) #capture l'image
    for ligne in range(min(pixh,len(image))):
        for colonne in range(min(pixl,len(image[0]))):
            temp = image[ligne][colonne]
            try:
                sortie[0][ligne*pixl+colonne] = temp[0] #on met en entrée à la suite les lignes de la photo dont on ne garde que la nuance de rouge, on rogne l'image si elle est trop grande, on remplace par du noir si trop petite$
            except IndexError: #si au lieu d'un tableau de trois nombres on a simplement un nombre
                sortie[0][ligne*pixl+colonne] = temp
    del image
    return reseau(poids,sortie)[-1]
    
def res(poids,nomim,attri):
    result = test(poids, nomim, attri)
    return attribut[result.index(max(result))]


##Algo génétique d'apprentissage: population de 40 réseaux. La génération suivante comprend le meilleur des réseaux et ce même réseau muté

import copy

def probfunc(x): #fonction donnant l'amplitude les mutations en fonctions des générations
    if x<=30:
        return 30
    else:
        return 30/(1+0.1*(x-30))+3

def mute(poids,nogeneration,probmutation = 0.5): #probmutation : probabilité qu'un poids mute d'une amplitude probfunc.
    lst = copy.deepcopy(poids) #deepcopy obligé pour eviter les alias avec poids...
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            for k in range(len(lst[i][j])):
                if random()<probmutation:
                    lst[i][j][k] += probfunc(nogeneration)*2*(random()-0.5)
    return lst
    
def gengeneration(meilleur,population,nogeneration): #meilleur=meilleur poids de la génération précedente
    generation = [meilleur]
    for i in range(1,population):
        nouveau = mute(meilleur,nogeneration)
        generation.append(nouveau)
    return generation
    
def apprentissage(fitvoulu = 1000, generationmax = 200): #generationmax = val de la gene max pr eviter boucle infinie si jamais EQM non atteint
    generation = [genpoidsalea() for i in range(population)]
    #On détermine quel poids donne le meilleur fitness
    nobestfit = 0 #membre de la pop avec le meilleur fitness
    bestfit = -1 #tous les fitness sont positifs ou nuls
    nogeneration = 0
    while bestfit<fitvoulu and nogeneration<=generationmax:
        nobestfit= 0
        bestfit = -1
        for nomembre in range(0,population):
            fitmembre,tauxreussite = fitness(generation[nomembre])
            if fitmembre > bestfit:
                besttauxreussite = tauxreussite
                bestfit = fitmembre
                nobestfit = nomembre
        generation = gengeneration(generation[nobestfit],population,nogeneration)
        print("Generation ",nogeneration, "Fitness: ", bestfit,"Taux reussite: ",int(besttauxreussite*10000)/100,"%" )
        nogeneration += 1
    return generation[0]
    
def fitness(poids):
    som = 0
    reussi = 0 #nb tst réussis
    for no,attri in enumerate(attribut):
        files = os.listdir(direc+"/"+attri) #cette liste contient les images pas encore testées
        for _ in range(nbtest):
            result = test(poids,files.pop(randint(0,len(files)-1)),attri) #on extrait aléatoirement un element de files pour le tester
            som += abs(sum(result)-1) #il faut que la somme soit égale à 1
            if result.index(max(result)) == no:
                reussi += 1
            else:
                som += 4 #on pénalise s'il ne trouve pas la bonne image
    fit = 1/som
    return (fit, reussi/(nbtest*len(attribut))) #fitness, taux de reussite
