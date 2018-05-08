
import matplotlib
matplotlib.use('Agg')
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import functions_bESN as fb
import bESN

def Sim(N, k, d, T, c, noise):
    return np.mean( fb.simulate(N=N, k=k, d=d, T=T, c = 0.5, noise = noise).entropy()[100:])


k_val = np.linspace(1, 300, 100)
n_val = np.linspace(0, 1.1, 100)
#d_val = np.linspace(-0.5, 0.5, 100)

x = k_val[:]
y = n_val[:]

#output = np.array([[ np.mean( fb.simulate(1000, k=k, d=0.15, T=300, c = 0.5, noise = n).entropy()[100:] ) for k in k_val] for n in tqdm(n_val)  ])

#np.mean( fb.simulate(1000, k=k, d=0.15, T=300, c = 0.5, noise = n).entropy()[100:] ) for k in k_val] for n in tqdm(n_val)

output = Parallel(-1)(delayed(Sim)(N = 1000, k=k, d=0.15, T=300, c = 0.5, noise = n)   for n in n_val for k in tqdm(k_val))



Var = np.array(output).reshape(len(y), len(x))


#Var= np.array( (len(n_val), len(k_val)) )
#for n  in tqdm(n_val):
 #   for k in k_val:
  #      a = np.mean(fb.simulate(1000, k=k, d=0.15, T=300, c = 0.5, noise = n).entropy()[100:])

z = Var[:,:]

np.save('KVSnoise', z)

plt.pcolor(x,y,z)
plt.xlabel("K")
plt.ylabel("noise")
#plt.title("Entropy plot")
plt.colorbar()
#plt.show()
plt.savefig("KVSnoise.pdf", format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

#plt.plot(x[1:], 1/sqrt(2*x[1:]), 'k--')
#plt.plot(x[1:], -1/sqrt(2*x[1:]), 'k--')