"""Idée : Apprentisage génétique d'un réseau de neurone de la fonction :  maintien du balais en l'air. Entrée : G(t), et dG(t), Sortie = S, on a ddxo(t) = (s-0.5)*10 (ces rapports sont dus au fait que la sortie est une valeur entre 0 et 1)
Fitness >=0 : 0 si la barre tombe durant l'un des nbtest essais, sinon la somme des valeurs de fitness de chaque tests stictement supérieures à 0 et aussi grandes que la base était stable, cad que la moyenne temporelle de la valeur absolue de ddxo + la moyenne temporelle de la valeur absolue de la vitesse est petite
La partie réseau de neurone est la même que le programme de base, en remplacant EQ, EQM, Loi et Test par fitness
Projet : mettre en entrée la position xo, ce qui est fait dans ce fichier"""

## IMPORTS

from random import *
from math import *
import matplotlib.pyplot as pt
import copy
from pylab import *
from matplotlib import animation

## RESEAU DE NEURONE

# 0 <= nochouche <= len(nbneurone) , et nocouche=0 correspond à l'entrée, nocouche = 1 correspond à la première couche de neurone
#poids = [liaison nocouche -> nochouche+1][noneurone de la couche nochouche+1][noentre]
#nbneurone[nocouche] = nbneurone de la couche nocouche, nbneurone[0] correspond donc au nombre d'entrées du réseau!!
#sortie = [nocouche][noneurone] correspond à la sortie du neurone. Pour, sortie[0], on parle donc des valeurs en entrée du réseau
nbneurone = [3,4,1] #la couche intermédiaire est encore à déterminer

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
import copy

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

def fitness(poids, nbneurone, nbit, dt): #lance une sinulation, détermine le fitness selon les règles en haut, nbit : nombre d'itérations, dt : intervalle de temps entre deux valeur, nbtest : nombre de test avec une valeur initiale de G différente
    fitness = 0
    T = [0.1,-0.1,0.2,-0.2,0.4,-0.4]
    for j in T:
        G = j
        dG = 0
        sum = 0 #on somme toutes les valeurs absolues de ddxo pour en déterminer la moyenne temporelle
        dxo = 0
        xo = 0
        for i in range(nbit):
            sortie = gensortie(nbneurone)
            sortie[0] = [G,dG,xo]
            ddxo = (reseau(poids, sortie, nbneurone)[-1][0]-0.5)*10 #valeur à t = i*dt de ddxo
            if G > 1.5707963267948966 or G < -1.5707963267948966: #si G inférieur à -pi/2 ou sup à pi/2, alors la barre est tombée, le fitness est de 0
                return 0
            sum += dt*abs(xo)
            (G,dG) = next(G,dG,ddxo,1,dt) #on prend une tige d'un mètre, on calcule les valeurs de G et dG à t = (i+1)*dt
            xo += dxo*dt
            dxo += ddxo*dt
        fitness += 1/sum #inversement proportionnel à la moyenne temporelle et strict supérieur à nbit
    return fitness
    
def apprentissage(nbneurone, population = 50, fitvoulu = 100, generationmax = 500, nbit = 100, dt = 1/10): #generationmax = val de la gene max pr eviter boucle infinie si jamais EQM non atteint
    generation = [genpoidsalea(nbneurone) for i in range(population)]
    #On détermine quel poids donne le meilleur fitness
    nobestfit = 0 #membre de la pop avec le meilleur fitness
    bestfit = -1 #tous les fitness sont positifs ou nuls
    nogeneration = 0
    X = []
    Y = []
    while bestfit<fitvoulu and nogeneration<=generationmax:
        nobestfit= 0
        bestfit = -1
        for nomembre in range(0,population):
            fitmembre = fitness(generation[nomembre], nbneurone, nbit, dt)   
            if fitmembre > bestfit:
                bestfit = fitmembre
                nobestfit = nomembre
        generation = gengeneration(generation[nobestfit],population)
        if nogeneration%10 == 0:
            print("Generation ",nogeneration, "Fitness : ", bestfit)
            X.append(nogeneration)
            Y.append(bestfit)
        nogeneration += 1
    pt.plot(X,Y)
    pt.show()
    return (generation[0], nogeneration, bestfit, nobestfit)
        
## SIMULATION DU BALAIS

#tige homogène de masse m. O-----M. pt O mobile sur l'axe x, de coord xo, et le pt M est à dist l de O, de coord x,y
#derivee première de x : dx, seconde : ddx
#Eq diff sur theta (que l'on note G) (angle que la barre fait avec Oy): ddG = 3/(2*l)*(g*sin(G)-ddxo*cos(G))
#dt : intervalle de temps sur lequel est calculé la prochaine valeur

def f(G,ddxo,l):
    return 3/(2*l)*(9.81*sin(G)-ddxo*cos(G))

def next(G,dG,ddxo,l,dt):
    k1 = f(G,ddxo,l)
    k2 = f(G + dt/2*dG,ddxo,l)
    k3 = f(G + dt/2*dG + (dt**2)/4*k1,ddxo,l)
    k4 = f(G + dt*dG + (dt**2)/2*k2,ddxo,l)
    return (G + dt*dG + (dt**2)/6*(k1 + k2 + k3), dG + dt/6*(k1 + 2*k2 + 2*k3 + k4)) #retourne D, dG
    
def integre(DDXO, dt, DXO0, XO0):
    XO = [XO0]
    DXO = [DXO0]
    for i in DDXO[0:-1:]:
        XO.append(XO[-1] + dt*DXO[-1])
        DXO.append(DXO[-1] + dt*i)
    return XO
    

#DDXO[i] est la valeur de ddxo à t = dt*i, de même pour Gl, DGl X, Y, et X0, G0 et DG0 les vals de G et DG à t=0
def calc(DDXO, G0, DG0, l, dt): #on calcule autant d'itérations que de valeurs dans DDXO
    Gl = [G0]
    DGl = [DG0]
    long = len(DDXO)
    for i in range(long-1):
        (a,b) = next(Gl[i],DGl[i],DDXO[i],l,dt)
        Gl.append(a)
        DGl.append(b)
    return Gl


def affiche(Gl, DDXO, l = 1, dt = 1/25, DXO0 = 0, XO0 = 0):
    XO = integre(DDXO,dt,DXO0,XO0)
    X = [l*sin(i) for i in Gl]
    Y = [l*cos(i) for i in Gl]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-2, 2))
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    def position(i):
        x = np.cumsum([XO[i],X[i]])
        y = np.cumsum([0,Y[i]])
        return (x, y)
    def init():
        line.set_data([],[])
        return line,
    def animate(i):
        line.set_data(position(i))
        return line,
    ani = animation.FuncAnimation(fig, animate, frames=len(X), interval=dt*1000, blit=True, init_func=init)
    plt.show()

def montre(poids, nbneurone, nbit, dt,G0):
    G = G0
    dG = 0
    xo = 0
    dxo = 0
    DDXO = []
    for i in range(nbit):
        sortie = gensortie(nbneurone)
        sortie[0] = [G,dG,xo]
        DDXO.append((reseau(poids, sortie, nbneurone)[-1][0]-0.5)*10)
        xo += dxo*dt
        dxo += DDXO[-1]*dt
        (G,dG) = next(G,dG,DDXO[-1],1,dt)
    return DDXO
    
def k(p,nbneurone,G0):
    DDXO = montre(p,nbneurone,1000,0.1,G0)
    Gl = calc(DDXO,G0,0,1,0.1)
    affiche(Gl,DDXO,1,0.1,0,0)
