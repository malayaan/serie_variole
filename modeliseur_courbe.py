import numpy as np
import matplotlib.pyplot as plt

class Modeliseur:
    def __init__(self):
        # resultats des calculs
        self.entrée = None
        self.sortie = None

        #fonction densité et perte
        self.fonction = None
        self.perte = None

        #données a modéliser et parametres de la loi
        self.param = None
        self.données = None

        #vecteur erreur
        self.erreur = None
        self.evolution = None

    #mise en place du modeliseur
    def installation(self, x, fonction, perte, param, données, evolution):
        self.entrée = x
        self.fonction = fonction
        self.perte = perte
        self.param = np.array(param)
        self.données = données
        self.evolution = evolution

    #calcul de la loi et mise a jour de la sortie
    def calcul_directe(self):
        self.sortie = self.fonction(self.entrée, self.param)

    #corrige les parametres
    def correction_erreur(self, learning_rate, gamma, epsilon):
        for i in range(len(self.param)):

            #calcul la dérivée partielle
            param_ajustés = self.param.copy()
            param_ajustés[i] += epsilon
            derivee_partiel= (self.perte(self.fonction(self.entrée,param_ajustés), self.données) - (self.perte(self.fonction(self.entrée,self.param), self.données)))/epsilon

            #mise à jours du modeliseur
            self.evolution[i] = gamma *  np.linalg.norm(self.evolution) + learning_rate * derivee_partiel
            self.param[i] -= self.evolution[i]
        self.calcul_directe()


    def entrainement(self, iterations, learning_rate, gamma, epsilon):
        # boucle d'entrainement
        for i in range(iterations):
            # corrige l'erreur
            self.correction_erreur( learning_rate, gamma, epsilon)

            # calcul l'erreur de l'affichage
            err = self.perte(self.données, self.sortie)

            # calcul de l'erreur moyenne sur l'échantillon
            #print('itération %d/%d   errreur=%f' % (i+1, iterations, err))
        print(self.param)

#fonctions
def normal(x ,param):
    mu, sigma, A = param[0], param[1], param[2]
    return A*(np.exp(-(x - mu)**2 / (2 * sigma**2))
       / (sigma * np.sqrt(2 * np.pi)))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

# main
if __name__ == "__main__":
    # mémoire
    x= np.linspace(1,len(n_malade),len(n_malade))
    n_malade=[3,1,2,2,1,1,3,3,5,2,5,1,3,5,6,6,10,7,3,5,12,16,8,6,20,13,17,15,23,27,22,21,21,22,24,28,40,42,42,42,41,94,79,74,98,113,94,91,93,143,139,139,182,121,100,269,169,241,234,282,157,138,340,310,315,296,359,181,192,587,312,470,401,439,260,217,661,446,521,507,470,264,202,563,487,502,450,443,228,192,688,403,389,440,435,207,160,555,471,303,384,368,126,125,498,346,273,493,336,115,96,128,561,202,247,253,97,77,281,242,199,183,166,72,63,302,187,232,385,143,48,31,159,150,140,120,97,22,6,90,81,60]
    n_malade=np.array(n_malade)/950

    #lissage des données
    y1= n_malade.copy()
    for i in range(len(n_malade)-6):
        y1[i+3]=np.mean(n_malade[i:i+6])
    y=y1.copy()
    for i in range(len(y1)-6):
        y[i+3]=np.mean(y1[i:i+6])
    """
    y= normal(x, [2,20,1])
    """
    #mise en place du modeliseur
    modeliseur = Modeliseur()
    modeliseur.installation(x, normal, mse, [0.9,0.9,0.9], y, [0,0,0])
    modeliseur.entrainement(100000, 4000, 0, 0.0001)

    #affichage des données
    plt.plot(x,y,'r')
    plt.plot(x,normal(x, modeliseur.param),'b')
    plt.show()