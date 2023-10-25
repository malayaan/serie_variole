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
        #print('itération %d/%d   errreur=%f' % (i+1, iterations, err))

# mémoire
n_malade=[1,0,1,4,0,1,2,2,4,0,3,3,6,5,9,6,1,3,12,7,8,4,13,12,9,13,19,18,15,20,16,15,13,17,22,34,36,28,19,58,59,62,69,60,46,47,43,90,84,83,122,70,55,186,180,182,201,233,126,69,286,265,288,241,346,167,86,520,365,466,329,403,205,137,607,437,510,534,454,164,146,783,498,649,399,490,210,151,789,390,484,416,152,99,923,636,349,466,331,90,47,560,412,252,576,375,49,25,29,742,273,291,248,29,26,301,218,78]
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
    y=np.array(decoupeur(n2_malade[6:],7))
    x_train=x[:int(2*len(x)/3)]
    y_train=y[:int(2*len(x)/3)]
    x_test=x[int(2*len(x)/3):]
    y_test=y[int(2*len(x)/3):]

    # Reseau
    net = Reseau()
    net.ajoute(Couche_Neurone_connectee(7, 90))
    net.ajoute(Couche_d_activation(relu, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(90, 80))
    net.ajoute(Couche_d_activation(tanh, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(80, 7))
    net.ajoute(Couche_d_activation(tanh, tanh_derivee))

    # train
    net.installation(mse, mse_derivee)
    net.entrainement(x_train, y_train, iterations=10000, learning_rate=0.001)

    # test
    y_result = net.prediction(x_test)
    y_result=[x.tolist()[-1] for x in y_result]

    #affichage des prédictions
    y0, y1, y2, y3, y4, y5, y6= [i[0] for i in y_result ],[i[1] for i in y_result ],[i[2] for i in y_result ], [i[3] for i in y_result ], [i[4] for i in y_result ], [i[5] for i in y_result ], [i[6] for i in y_result ]
    fig, axs = plt.subplots(3, 3)
    n= int(2*len(x)/3)
    axs[0, 0].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[0, 0].bar(np.linspace(n,len(y0)+n-1,(len(y0))), y0, alpha=0.7)
    axs[0, 0].set_title('1 jours d\'avance')
    n+=1
    axs[0, 1].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[0, 1].bar(np.linspace(n,len(y1)+n-1,(len(y1))), y1, alpha=0.7)
    axs[0, 1].set_title('2 jours d\'avance')
    n+=1
    axs[0, 2].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[0, 2].bar(np.linspace(n,len(y2)+n-1,(len(y2))), y2, alpha=0.7)
    axs[0, 2].set_title('3 jours d\'avance')
    n+=1
    axs[1, 0].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[1, 0].bar(np.linspace(n,len(y3)+n-1,(len(y3))), y3, alpha=0.7)
    axs[1, 0].set_title('4 jours d\'avance')
    n+=1
    axs[1, 1].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[1, 1].bar(np.linspace(n,len(y3)+n-1,(len(y3))), y4, alpha=0.7)
    axs[1, 1].set_title('5 jours d\'avance')
    n+=1
    axs[1, 2].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[1, 2].bar(np.linspace(n,len(y3)+n-1,(len(y3))), y5, alpha=0.7)
    axs[1, 2].set_title('6 jours d\'avance')
    n+=1
    axs[2, 1].bar(np.linspace(0,len(n2_malade)-1,(len(n2_malade))),n2_malade,alpha=1)
    axs[2, 1].bar(np.linspace(n,len(y3)+n-1,(len(y3))), y6, alpha=0.7)
    axs[2, 1].set_title('7 jours d\'avance')
    plt.show()