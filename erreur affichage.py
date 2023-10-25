from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

#data
n_malade=[3,1,2,2,1,1,3,3,5,2,5,1,3,5,6,6,10,7,3,5,12,16,8,6,20,13,17,15,23,27,22,21,21,22,24,28,40,42,42,42,41,94,79,74,98,113,94,91,93,143,139,139,182,121,100,269,169,241,234,282,157,138,340,310,315,296,359,181,192,587,312,470,401,439,260,217,661,446,521,507,470,264,202,563,487,502,450,443,228,192,688,403,389,440,435,207,160,555,471,303,384,368,126,125,498,346,273,493,336,115,96,128,561,202,247,253,97,77,281,242,199,183,166,72,63,302,187,232,385,143,48,31,159,150,140,120,97,22,6,90,81,60]
n_malade=np.array(n_malade)/950
x= np.linspace(1,len(n_malade),len(n_malade))
y1= n_malade.copy()
A = np.mean(n_malade)*len(n_malade)
#lissage des données
for i in range(len(n_malade)-6):
    y1[i+3]=np.mean(n_malade[i:i+6])

y=y1.copy()

for i in range(len(y1)-6):
    y[i+3]=np.mean(y1[i:i+6])

#defining the functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def lognormal(x, param):
    mu, sigma, A = float(param[0]), float(param[1]), np.mean(n_malade)*len(n_malade)
    return A/(x*sigma*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

# defining surface and axes
MU = np.outer(np.array([mu for mu in np.arange(0,10,10/99)]), np.ones(100))
SIGMA1=[sigma for sigma in np.arange(0.1,1,0.9/99)]
SIGMA = SIGMA1.copy()
A=[]
i=0
for mu in MU:
    A+=[[]]
    i+=1
    for sigma in SIGMA:
        A[-1].append(mse(lognormal(x, [float(mu[-1]),sigma]), y))
print(A)
fig = plt.figure()
 
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(MU, SIGMA, np.array(A), cmap ='viridis')
ax.set_title('plot de l\'erreur')
plt.show()