""" importations de bibliotheque"""
import numpy as np
import matplotlib.pyplot as plt

# création des classes
# première classe
class Couche:
    def __init__(self):
        self.entrée = None
        self.sortie = None

    # calcul la sortie d'une couche en fonction de l'entrée
    def propagataion_directe(self, entrée):
        raise Rien_d_implémenté

    # calcul dE/dX pour un certain dE/dY
    def retro_propagataion(self, erreur_sortie, learning_rate):
        raise Rien_d_implémenté

# hérité de la classe couche
class Couche_Neurone_connectee(Couche):
    # entrée_taille = nombre de neurone en entree, sortie_taille = nombre de neurone en sortie
    def __init__(self, entrée_taille, sortie_taille):
        self.poids = np.random.rand(entrée_taille, sortie_taille) - 0.5
        self.biais = np.random.rand(1, sortie_taille) - 0.5

    # renvoie la sortie pour une certaine entrée
    def propagataion_directe(self, entrée_donnée):
        self.entrée = entrée_donnée
        self.sortie = np.dot(self.entrée, self.poids) + self.biais
        return self.sortie

    # calcul dE/dW, dE/dBpour une erreur_sortie=dE/dY. renvoie erreur_entrée=dE/dX.
    def retro_propagataion(self, erreur_sortie, learning_rate):
        erreur_entrée = np.dot(erreur_sortie, self.poids.T)
        poids_erreurs = np.dot(self.entrée.T, erreur_sortie)
        # biais = erreur_sortie

        # mise a jour des parametres
        self.poids -= learning_rate * poids_erreurs
        self.biais -= learning_rate * erreur_sortie
        return erreur_entrée

# hérité de la classe couche
class Couche_d_activation(Couche):
    def __init__(self, activation, activation_derivee):
        self.activation = activation
        self.activation_derivee = activation_derivee

    def retro_propagataion(self, erreur_sortie, learning_rate):
        return self.activation_derivee(self.entrée) * erreur_sortie

    def propagataion_directe(self, entrée_donnée):
        self.entrée = entrée_donnée
        self.sortie = self.activation(self.entrée)
        return self.sortie

class Reseau:
    def __init__(self):
        self.Couches = []
        self.perte = None
        self.perte_derivee = None

    # ajoute Couche au Reseau
    def ajoute(self, Couche):
        self.Couches.append(Couche)

    # installe la focntion perte dans le réseau
    def installation(self, perte, perte_derivee):
        self.perte = perte
        self.perte_derivee = perte_derivee

    # donne la prédiction de la sortie pour une entrée donnée
    def prediction(self, entrée_donnée):
        # taille de l'échabntillon
        résultat = []
        échantillonnage = len(entrée_donnée)
        # essai du réseau sur tout l'échantillon
        for i in range(échantillonnage):
            # propagation directe
            sortie = entrée_donnée[i]
            for Couche in self.Couches:
                sortie = Couche.propagataion_directe(sortie)
            résultat.append(sortie)
        return résultat

    # entrainement du réseau
    def entrainement(self, x_train, y_train, iterations, learning_rate):
        # dimention de l'échantillon
        échantillonnage = len(x_train)

        # boucle d'entrainement
        for i in range(iterations):
            err = 0
            for j in range(échantillonnage):
                # propagation directe
                sortie = x_train[j]
                for Couche in self.Couches:
                    sortie = Couche.propagataion_directe(sortie)

                # calcul l'erreur de l'affichage
                err += self.perte(y_train[j], sortie)

                # retro_propagataion
                errreur = self.perte_derivee(y_train[j], sortie)
                for Couche in reversed(self.Couches):
                    errreur = Couche.retro_propagataion(errreur, learning_rate)

            # calcul de l'erreur moyenne sur l'échantillon
            err /= échantillonnage
        print('itération %d/%d   errreur=%f' % (i+1, iterations, err))

# mémoire
n_malade=[1,1,2,0,0,1,4,3,1,2,1,1,1,2,3,4,1,4,0,4,5,6,6,9,6,3,5,12,17,6,4,20,13,17,14,23,28,20,22,22,24,24,26,40,42,43,43,47,94,80,76,99,118,102,100,106,153,157,166,198,151,139,299,212,273,263,345,206,174,373,309,315,301,372,250,230,532,355,471,428,461,312,261,569,507,488,450,494,346,269,552,462,517,429,432,304,262,556,379,370,440,434,282,200,429,414,322,358,364,175,173,405,328,277,348,334,151,122,170,421,237,254,253,117,93,198,236,173,185,198,110,91,292,183,152,160,168,74,59,171,142,117,127,113,64,39,129,110,83,104,106,33,25,47,129,88,82,80,21,14,92,47,71,72,28,12,7,62,22,6]
n2_malade=[x/max(n_malade) for x in n_malade]

# fonctions
# fonction perte et sa dérivée:  squared error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))
def mse_derivee(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

#fonctions d'activations
def tanh(x):
    return np.tanh(x);
def tanh_derivee(x):
    return 1-np.tanh(x)**2;
def douce(x):
    return np.log(1+np.exp(x))
def douce_derivee(x):
    return 1/(1+np.exp((-x)))
def relu(x):
    return (x>0)*x
def relu_derivee(x):
    return x>0

#création de la base de donnée
def decoupeur(x,n):
    y=[[x[:n]]]
    for i in x[n:]:
        z=y[-1][-1][1:]+[i]
        if len(z)==n:
            y+=[[z]]
    return y

# main
if __name__ == "__main__":

    #préparation des donnees(n_malade)
    x=np.array(decoupeur(n2_malade,7))
    y=np.array(n2_malade)
    x_train=x[:58]
    y_train=y[:58]
    x_test=x[112:]
    y_test=y[112:]

    # Reseau
    net = Reseau()
    net.ajoute(Couche_Neurone_connectee(7, 90))
    net.ajoute(Couche_d_activation(relu, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(90, 80))
    net.ajoute(Couche_d_activation(tanh, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(80, 1))
    net.ajoute(Couche_d_activation(tanh, tanh_derivee))

    # train
    net.installation(mse, mse_derivee)
    net.entrainement(x_train, y_train, iterations=500, learning_rate=0.1)
    # test
    y_result = net.prediction(x_test)
    y_result=[x.tolist()[-1][-1] for x in y_result]
    #affichage des prédictions
    n= 112
    plt.bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    plt.bar(np.linspace(119,169,5),y_result,alpha=0.7)

    plt.show()