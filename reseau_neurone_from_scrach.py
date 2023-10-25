""" importations de bibliotheque"""


import numpy as np



""" création des classes"""

# classe de base
class Couche:
    def __init__(self):
        self.entrée = None
        self.sortie = None

    # computes the output Y of a layer for a given input X
    def  propagation_direct(self, entrée):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def retro_propagationn(self, erreur_sortie, learning_rate):
        raise NotImplementedError


# classe hérité de la classe Couche
class Couche_Neurone_connectee(Couche):
    # taille_entree = nombre de neurones en entrée
    # taille_sortie = nombre de neurones en sortie
    def __init__(self, taille_entree, taille_sortie):
        self.poids = np.random.rand(taille_entree, taille_sortie) - 0.5
        self.biais = np.random.rand(1, taille_sortie) - 0.5

    # renvoie la sorti pour une entrée donnée
    def propagation_direct(self, donnée_entrée):
        self.entrée = donnée_entrée
        self.sortie = np.dot(self.entrée, self.poids) + self.biais
        return self.sortie

    # calcule dE/dW, dE/dB pour une certaine erreur_sortie=dE/dY. Renvoie erreur_entrée=dE/dX.
    def retro_propagation(self, erreur_sortie, learning_rate):
        erreur_entrée = np.dot(erreur_sortie, self.poids.T)
        erreur_poid = np.dot(self.entrée.T, erreur_sortie)
        # dbiais = erreur_sortie

        # mise a jour des paramètres
        self.poids -= learning_rate * erreur_poid
        self.biais -= learning_rate * erreur_sortie
        return erreur_sortie


# classe hérité de la classe Couche
class Couche_activation(Couche):
    def __init__(self, activation, activation_dérivée):
        self.activation = activation
        self.activation_dérivée = activation_dérivée

    # utilise la fonction activation
    def propagation_direct(self, donnée_entrée):
        self.entrée = donnée_entrée
        self.sortie = self.activation(self.entrée)
        return self.sortie

    # renvoie erreur_entrée=dE/dX pour une certaine erreur_sortie=dE/dY.
    def retro_propagation(self, erreur_sortie, learning_rate):
        return self.activation_dérivée(self.entrée) * erreur_sortie


#classe du réseau de neurone
class Reseau:
    def __init__(self):
        self.couches = []
        self.perte = None
        self.perte_dérivée = None

    # ajoute une couche au réseau
    def ajoute(self, couche):
        self.couches.append(couche)

    # met en place la fonction perte
    def set_up_perte(self, perte, perte_dérivée):
        self.perte = perte
        self.perte_dérivée = perte_dérivée

    # prédit le résultat pour les vecteur donnée en entrée
    def prediction(self, entrée):
        # sample dimension first
        samples = len(entrée)
        resultats = []

        # fait tourner le réseau pour tout les éléments en entré
        for i in range(samples):
            # forward propagation
            sorti = entrée[i]
            for couche in self.couches:
                sorti = couche.propagation_direct(sorti)
            resultats.append(sorti)

        return resultats


    # entrainement du réseau
    def adapatation(self, x_train, y_train, itération, learning_rate):
        #taille de la base d'entrainement
        samples = len(x_train)

        #boucle d'entrainement
        for i in range(itération):
            err = 0
            for j in range(samples):
                #propagation directe
                sorti = x_train[j]
                for couche in self.couches:
                    sorti = couche.propagation_direct(sorti)

                # calcul de la perte d'information pour l'affichage
                err += self.perte(y_train[j], sorti)

                # rétropropagation
                error = self.perte_dérivée(y_train[j], sorti)
                for couche in reversed(self.couches):
                    error = couche.retro_propagation(error, learning_rate)

            # calcul de l'erreur moyenne sur l'ensemble des essais
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, itération, err))



"""fonction"""

# fonction perte et sa dérivée: mean squared error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_derivee(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

#fonction d'activation tangeante
def tanh(x):
    return np.tanh(x);

def tanh_derivee(x):
    return 1-np.tanh(x)**2;

"""main"""

if __name__ == "__main__":

    # training data pour un xor
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # mise en place du réseau
    reseau = Reseau()
    reseau.ajoute(Couche_Neurone_connectee(2, 3))
    reseau.ajoute(Couche_activation(tanh, tanh_derivee))
    reseau.ajoute(Couche_Neurone_connectee(3, 1))
    reseau.ajoute(Couche_activation(tanh, tanh_derivee))

    # train
    reseau.set_up_perte(mse, mse_derivee)
    reseau.adapatation(x_train, y_train, itération=1000, learning_rate=0.1)