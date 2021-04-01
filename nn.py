import random
import math
import matplotlib.pyplot as pt
#Réseau de neurone dont l'apprentissage est fait par un algorithme génétique (séléction, mutation), où une seule liste de poids sera seléctionnée parmis une population de 100 listes de poids
#Declaration/Capture de variables
nbiterations = 300 #int(input("Saisissez le nombre d'itérations : "))
noiteration = 0
nbtests = 4 #nombre de tests sur un chaque membres pour déterminer la somme des EQM
notest = 0
nbmembres = 100 #int(input("Saisissez la taille des populations : ")) #donne le nombre de membres dans les populations
nomembre = 0 #numéro du membre courant (de 0 à nbmembres-1)
seuilEQM = 0 #float(input("Saisissez le seuil EQM en dessous duquel les itérations s'arrêtent : "))
bestEQM = 9999 #Valeur de la plus faible erreur obtenue par un membre de la population (intialisée a une forte valeur pour retenir les valeur plus faibles
membrebestEQM = 0 #numero du membre ayant obtenu la plus faible erreur
nbcouches = 2 #int(input("Saisissez le nombre de couches de neurones : ")) #nombre de couches de neurones, sans compter la couche d'entrees qui est comptee dans nbneurones
nocouche = 0 #numero couche courante (de 0 à nbcouches) (les entrees du reseau sont considérées comme la couche 0)
noneurone = 0 #numero neurone courant (de 0 à nbneurones[nocouche] - 1)
noneuronesuiv = 0 #numero neurone couche suivante
noentree = 0 #numero entree courante (de 0 à nbneurones[nocouche - 1] - 1), correspond donc à un neurone de la couche précédente
ans = 0 #Variable temporaire

nbneurones = [2,2,1] #Donne le nombre de neurones dans la couche souhaitée : nbneurones[nocouche]
"""nbneurones[0] = int(input("Saisissez le nombre d'entrees : "))
for nocouche in range(1, nbcouches+1):
    print("Saisissez le nombre de neurones de la couche ", nocouche, " : ")
    nbneurones.append(int(input()))"""
EQM = [0]*nbmembres #Donne la somme des Erreur Quadratique Moyenne d'une itération pour chaque membre : EQM[nomembre]


#Construction de la liste poids, donnant la liste des poids pour chaque membres d'une population poids[nomembre][nocouche][noneurone][noentree]
poids = []
for nomembre in range(0,nbmembres):
    poids.append([[]])
    for nocouche in range(1, nbcouches+1):
        poids[nomembre].append([])
        for noneurone in range(nbneurones[nocouche]):
            poids[nomembre][nocouche].append([])
            for noentree in range(nbneurones[nocouche-1]+1): #le nombre d'entrees d'un neurone correspond au nombre de neurones +1 dans la couche précédente à cause du coëfficient de biais en fin de la liste (poids[nbneurones[nocouche-1]]
                poids[nomembre][nocouche][noneurone].append(2*random.random()-1)
                
                
outneurone = [[0]*nbneurones[nocouche] for nocouche in range(nbcouches+1)] #Donne les sorties de chaque neurone ou les entrees lorsque nocouche = 0 et les sorties du reseau entier pour nocouche = nbcouches, outneurone[nocouche][noneurone]
gradient = [[0]*nbneurones[nocouche] for nocouche in range(nbcouches+1)] #Donne le gradient local de chaque neurone, gradient[nocouche][noneurone]
erreur = [0]*nbneurones[nbcouches] #Donne l'erreur de chaque neurone de la couche de sortie, erreur[noneurone], où la couche courante est la couche de sortie, donc nocouche = nbcouche
outvoulu = [0]*nbneurones[nbcouches] #Donne la sortie désirée pour l'exemple traité pour chaque neurone de la couche de sortie, outvoulu[noneurone]
list = [[0,0,0],[1,1,0],[0,1,1],[1,0,1]] #Entrées et resultats de la fonction XOR
X = []
Y = []
for noiteration in range(nbiterations):
    for notest in range(nbtests):
        #Construction d'un exemple d'apprentissage de la fonction XOR
        outneurone[0][0] = list[notest][0]
        outneurone[0][1] = list[notest][1]
        outvoulu[0] = list[notest][2]
        #Fin construction exemple
        for nomembre in range(0,nbmembres):
            #Propagation des entrées vers l'avant du réseau
            for nocouche in range(1, nbcouches+1):
                for noneurone in range(0, nbneurones[nocouche]):
                    ans = 0
                    for noentree in range(0, nbneurones[nocouche-1]): #somme des entrees multipliées par leur poids
                        ans += outneurone[nocouche-1][noentree]*poids[nomembre][nocouche][noneurone][noentree]
                    ans += poids[nomembre][nocouche][noneurone][nbneurones[nocouche-1]] #on rajoute le coëfficient de biais
                    try:
                        outneurone[nocouche][noneurone] = 1/(1+math.exp(-ans)) #on passe par la fonction non linéaire sigmoide et on le met dans la liste outneurone
                    except OverflowError: #il se peut que ans soit excessivement grand pour les fonctions
                        if ans > 0:
                            outneurone[nocouche][noneurone] = 1.0
                        else:
                            outneurone[nocouche][noneurone] = 0.0
            #Fin propagation
            #Calcul de la somme des Erreurs Quadratiques Moyennes
            ans = 0
            for noneurone in range(0, nbneurones[nbcouches]):
                erreur[noneurone] = outvoulu[noneurone]-outneurone[nbcouches][noneurone] #on calcule l'erreur de chaque sortie
                ans += pow(erreur[noneurone],2) #calcul de la somme des erreurs quadratiques pour calculer l'erreur quadratique moyenne
            if notest == 0:
                EQM[nomembre] = math.sqrt(ans) #Lorsque l'on est au premier test, on réinitialise les valeurs de la liste
            else:
                EQM[nomembre] += math.sqrt(ans)
                if notest == nbtests-1:
                    EQM[nomembre] /= nbtests
            
            #Fin calcul EQM
    #Selection du meilleur membre, sauvegarde de sa liste de poids
    bestEQM = 99
    for nomembre in range(0, nbmembres):
        if EQM[nomembre] < bestEQM:
            bestEQM = EQM[nomembre]
            membrebestEQM = nomembre
    if bestEQM <= seuilEQM:
        break
    if membrebestEQM != 0:
        poids[0] = poids[membrebestEQM] #On enregistre la liste des poids du meilleur membre sous le numéro de membre 1
    X.append(noiteration)
    Y.append(bestEQM)
    #Fin de selection du meilleur membre
    #Mutation des poids
    for nomembre in range(1, nbmembres):
        for nocouche in range(1, nbcouches+1):
            for noneurone in range(0, nbneurones[nocouche]):
                for noentree in range(0, nbneurones[nocouche-1]+1):
                    poids[nomembre][nocouche][noneurone][noentree] = poids[0][nocouche][noneurone][noentree] + 100*(random.random()-0.5) #on modifie aleatoirement la valeur des poids (sauf celles des poids du membre #0, qui est une sauvegarde du meilleur précédent)
pt.plot(X,Y)
pt.show()
print (noiteration,poids[membrebestEQM])