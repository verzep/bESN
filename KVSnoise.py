import functions_bESN as fb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import bESN


import functions_bESN as fb


k_val = np.linspace(0.1, 300, 100)
n_val = np.linspace(0, 1.1, 100)
d_val = np.linspace(-0.5, 0.5, 100)
Var = np.array([[ np.mean( fb.simulate(1000, k=k, d=0.15, T=300, c = 0.5, noise = n).entropy()[100:] ) for k in k_val] for n in tqdm(n_val)  ])


x = k_val[:]
y = n_val[:]
z = Var[:,:]

plt.pcolor(x,y,z)
plt.xlabel("K")
plt.ylabel("noise")
#plt.title("Entropy plot")
plt.colorbar()
#plt.show()
plt.savefig("KVSnoise.pdf", format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

#plt.plot(x[1:], 1/sqrt(2*x[1:]), 'k--')
#plt.plot(x[1:], -1/sqrt(2*x[1:]), 'k--')