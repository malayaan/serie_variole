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
EE0, EE1, EE2, EE3, EE4, EE5, EE6= [], [], [], [], [], [], []
nbre_de_point =40
debut =1
fin =2000
n=5
for k in range(n):
    E0, E1, E2, E3, E4, E5, E6= [], [], [], [], [], [], []
    for i in range( debut, fin, int((fin-1-debut)/nbre_de_point) ):
        # Reseau
        net = Reseau()
        net.ajoute(Couche_Neurone_connectee(7, 3))
        net.ajoute(Couche_d_activation(tanh, tanh_derivee))
        net.ajoute(Couche_Neurone_connectee(3, 3))
        net.ajoute(Couche_d_activation(tanh, tanh_derivee))
        net.ajoute(Couche_Neurone_connectee(3, 3))
        net.ajoute(Couche_d_activation(tanh, tanh_derivee))
        net.ajoute(Couche_Neurone_connectee(3, 3))
        net.ajoute(Couche_d_activation(relu, tanh_derivee))
        net.ajoute(Couche_Neurone_connectee(3, 3))
        net.ajoute(Couche_d_activation(relu, tanh_derivee))
        net.ajoute(Couche_Neurone_connectee(3, 3))
        net.ajoute(Couche_d_activation(relu, tanh_derivee))
        net.ajoute(Couche_Neurone_connectee(3, 7))
        net.ajoute(Couche_d_activation(tanh, tanh_derivee))

        # train
        net.installation(mse, mse_derivee)
        net.entrainement(x_train, y_train, iterations=200, learning_rate=i*0.001)

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
        E3+=[mse(test, y4)]
        E4+=[mse(test, y4)]
        E5+=[mse(test, y5)]
        E6+=[mse(test, y6)]
        print("i=", i)
    print("k=",k)

    EE0 += [np.array(E0)]
    EE1 += [np.array(E1)]
    EE2 += [np.array(E2)]
    EE3 += [np.array(E3)]
    EE4 += [np.array(E4)]
    EE5 += [np.array(E5)]
    EE6 += [np.array(E6)]

EEE0=EE0[0]
for i in range(1,n):
    EEE0+=E0[i]
EEE0/=n
EEE1=EE1[0]
for i in range(1,n):
    EEE1+=E1[i]
EEE1/=n
EEE2=EE2[0]
for i in range(1,n):
    EEE2+=E2[i]
EEE2/=n
EEE3=EE3[0]
for i in range(1,n):
    EEE3+=E3[i]
EEE3/=n
EEE4=EE4[0]
for i in range(1,n):
    EEE4+=E4[i]
EEE4/=n
EEE5=EE5[0]
for i in range(1,n):
    EEE5+=E5[i]
EEE5/=n
EEE6=EE6[0]
for i in range(1,n):
    EEE6+=E6[i]
EEE6/=n

plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE0, label='0')
plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE1, label='1')
plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE2, label='2')
plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE3, label='3')
plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE4, label='4')
plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE5, label='5')
plt.plot(np.linspace(debut, fin-1,nbre_de_point+1), EEE6, label='6')
plt.legend()
plt.title("erreur en fonction de la taille du réseau")
plt.show()