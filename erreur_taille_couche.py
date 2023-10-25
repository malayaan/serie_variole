
import os
os.chdir(r"C:\Users\decroux paul\Documents\info\python")
from neurone_net import *
import numpy as np

#préparation des donnees(n_malade)
x=np.array(decoupeur(n2_malade,7))
y=np.array(decoupeur(n2_malade[6:],7))
x_train=x[:int(2*len(x)/3)]
y_train=y[:int(2*len(x)/3)]
x_test=x[int(2*len(x)/3):]
y_test=y[int(2*len(x)/3):]

#stockage des erreurs
E0, E1, E2, E3, E4, E5, E6= [], [], [], [], [], [], []
nbre_de_point =2
debut =1
fin =152
for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) ):

    # Reseau
    net = Reseau()
    net.ajoute(Couche_Neurone_connectee(7, i))
    net.ajoute(Couche_d_activation(relu, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(i, i))
    net.ajoute(Couche_d_activation(relu, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(i, i))
    net.ajoute(Couche_d_activation(relu, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(i, i))
    net.ajoute(Couche_d_activation(relu, tanh_derivee))
    net.ajoute(Couche_Neurone_connectee(i, 7))
    net.ajoute(Couche_d_activation(tanh, tanh_derivee))

    # train
    net.installation(mse, mse_derivee)
    net.entrainement(x_train, y_train, iterations=1000, learning_rate=0.1)

    # test
    y_result = net.prediction(x_test)
    y_result=[x.tolist()[-1] for x in y_result]

    #affichage des prédictions
    y0, y1, y2, y3, y4, y5, y6= [i[0] for i in y_result ],[i[1] for i in y_result ],[i[2] for i in y_result ], [i[3] for i in y_result ], [i[4] for i in y_result ], [i[5] for i in y_result ], [i[6] for i in y_result ]
    #print(len(y0), len(y1), len(y2))
    test=np.array(n2_malade[int(2*len(n2_malade)/3)+2:])
    E0+=[mse(test, y0)]
    E1+=[mse(test, y1)]
    E2+=[mse(test, y2)]
    E3+=[mse(test, y3)]
    E4+=[mse(test, y4)]
    E5+=[mse(test, y5)]
    E6+=[mse(test, y6)]

    print(i)

plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E0, label='0')
plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E1, label='1')
plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E2, label='2')
plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E3, label='3')
plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E4, label='4')
plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E5, label='5')
plt.plot([i for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) )], E6, label='6')
plt.legend()
plt.title("erreur en fonction du nombre de neurone par couche")
plt.show()