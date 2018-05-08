'''Various functions for the analysis'''


import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from bESN import *


def ham_dis(x1, x2): #hamming distance
    return sum(el1 != el2 for el1, el2 in zip(x1, x2))

def H_b(x): #binary entropy
    p = np.mean([x == 1])
    
    q = 1 - p
    
    if p == 1 or p == 0:
        return 0
    
    return -p* np.log2(p) - q*np.log2(q)



def simulate(N, k, d, T=500, signal = None, c = 0.5, noise = 0):
    
    nw = bESN (N, k, d, c=c, noise= noise)
    nw.evolve(signal = signal)
    
    return nw
    
def  MC(signal, prediction):
    C = np.cov(signal, prediction)
    
    if C[0][1] == 0 or C[0][0] == 0 or C[1][1]== 0:
        return 0
    else:    
        return C[0][1]* C[0][1] / (C[0][0] * C[1][1])   

def simulate_perturbation(N, k, d, T, nPer, c = 0.5, noise = 0):
    nw = bESN (N, k, d, c=c, noise= noise)

    nw.evolve(nSteps = T)

    h_dist = []
    ener = []
    ent = []
    act = []



    for r in np.random.choice(N, nPer, replace = False):
        #print "r ",r
        nw_pert = nw.perturb(r)

        nw_pert.evolve(nSteps = T)

        ener.append(nw_pert.energy)
        act.append(nw_pert.activity)
        ent.append(nw_pert.entropy())
        
        dist = []
        
        for t in range(0,T+1):
            #print nw_pert.memory[t] == nw.memory[t] 
            #print ham_dis(nw_pert.memory[t], nw.memory[t])
            
            dist.append(float(ham_dis(nw_pert.memory[t], nw.memory[t])) / N)
        
        
        h_dist.append(dist)
        
    plt.figure(figsize=(14,3))
    
    pD = plt.subplot(1,4,1)
    
    Dm = np.array(h_dist).mean(axis=0)
    Dv = np.array(h_dist).std(axis=0)
    
    pD.plot(Dm, 'b' )
    pD.plot(Dm+Dv, 'r--')
    pD.plot(Dm-Dv, 'r--')
    pD.plot()
    pD.set_ylim([0,1])
    plt.title('average Hamming distance')

    pE = plt.subplot(1,4,2)
    
    Em = np.array(ener).mean(axis=0)
    Ev = np.array(ener).std(axis=0)
    
    pE.plot(Em, 'b' )
    pE.plot(Em+Ev, 'r--')
    pE.plot(Em-Ev, 'r--')
    pE.plot()
    pE.set_ylim([-1,1])
    plt.title('Energy')
    
    pA = plt.subplot(1,4,3)
   
    Am = np.array(act).mean(axis=0)
    Av = np.array(act).std(axis=0)
    
    pA.plot(Am, 'b' )
    pA.plot(Am+Av, 'r--')
    pA.plot(Am-Av, 'r--')
    pA.plot()
    pA.set_ylim([0,1])
    plt.title('Activity')
    
    pS =plt.subplot(1,4,4)
       
    Sm = np.array(ent).mean(axis=0)
    Sv = np.array(ent).std(axis=0)
    
    pS.plot(Sm, 'b' )
    pS.plot(Sm+Sv, 'r--')
    pS.plot(Sm-Sv, 'r--')
    pS.plot()
    pS.set_ylim([0,1])
    plt.title('Entropy')
    
    plt.show()


def simulate_perturbation_signal(N, k, d, signal, nPer, c = 0.5):
    nw = bESN (N, k, d, c=c, noise= noise)

    nw.evolve(signal = signal)
    h_dist = []
    ener = []
    ent = []
    act = []



    for r in np.random.choice(N, nPer, replace = False):
        #print "r ",r
        nw_pert = nw.perturb(r)

        nw_pert.evolve(signal = signal)

        ener.append(nw_pert.energy)
        act.append(nw_pert.activity)
        ent.append(nw_pert.entropy())
        
        dist = []
        
        for t in range(0, len(signal) +1):
            #print nw_pert.memory[t] == nw.memory[t] 
            #print ham_dis(nw_pert.memory[t], nw.memory[t])
            
            dist.append(float(ham_dis(nw_pert.memory[t], nw.memory[t])) / N)
        
        
        h_dist.append(dist)
        
    plt.figure(figsize=(14,3))
    
    pD = plt.subplot(1,4,1)
    
    Dm = np.array(h_dist).mean(axis=0)
    Dv = np.array(h_dist).std(axis=0)
    
    pD.plot(Dm, 'b' )
    pD.plot(Dm+Dv, 'r--')
    pD.plot(Dm-Dv, 'r--')
    pD.plot()
    pD.set_ylim([0,1])
    plt.title('average Hamming distance')

    pE = plt.subplot(1, 4, 2)
    
    Em = np.array(ener).mean(axis=0)
    Ev = np.array(ener).std(axis=0)
    
    pE.plot(Em, 'b')
    pE.plot(Em+Ev, 'r--')
    pE.plot(Em-Ev, 'r--')
    pE.plot()
    pE.set_ylim([-1, 1])
    plt.title('Energy')
    
    pA = plt.subplot(1,4,3)
   
    Am = np.array(act).mean(axis=0)
    Av = np.array(act).std(axis=0)
    
    pA.plot(Am, 'b' )
    pA.plot(Am+Av, 'r--')
    pA.plot(Am-Av, 'r--')
    pA.plot()
    pA.set_ylim([0,1])
    plt.title('Activity')
    
    pS =plt.subplot(1,4,4)
       
    Sm = np.array(ent).mean(axis=0)
    Sv = np.array(ent).std(axis=0)
    
    pS.plot(Sm, 'b' )
    pS.plot(Sm + Sv, 'r--')
    pS.plot(Sm - Sv, 'r--')
    pS.plot()
    pS.set_ylim([0,1])
    plt.title('Entropy')
    
    plt.show()

def compute_MC (N, k, d, signal):
    
    A  = bESN(N, k, d)
    A.evolve(signal)
    MC = []
    
    for delay in range(0,100, 10):
        mse = A.learn_signal(signal, delay= delay, 
                         #make_plot= True, 
                         test_length= 500)
        MC.append(A.MC)
    
    return  np.sum(MC)

def compute_MC_short (N, k, d, signal):
    
    A  = bESN(N, k, d)
    A.evolve(signal)
    MC = []
    
    for delay in range(0,10):
        mse = A.learn_signal(signal, delay= delay, 
                         #make_plot= True, 
                         test_length= 500)
        MC.append(A.MC)
    
    return  np.sum(MC)

def compute_MC_instant (N, k, d, signal, delay = 0, make_plot = False):
    
    A  = bESN(N, k, d)
    A.evolve(signal)
    MC = []

    mse = A.learn_signal(signal, delay= delay, 
                     make_plot= make_plot, 
                     test_length= 500)
    MC.append(A.MC)
    if make_plot:
        ene = plot(A.energy[:], label = "E")
        act = plot(A.activity[:], label = "A")
        ENT = A.entropy()
        ent =plot(ENT[:], label = "S")
        plt.legend()
        plt.show()
    
    return  np.sum(MC)


def compute_MC_multi (N, k, d, signal):
    A  = bESN(N, k, d)
    A.evolve(signal)
    MC = []
    
    for delay in range(0,10):
        mse = A.learn_signal(signal, delay= delay, 
                         #make_plot= True, 
                         test_length= 500)
        MC.append(A.MC)
    
    return MC

def MC_multi_exp(signal):

    N = 1000    
    d_val = np.linspace(0, 0.20, 20)
    k_val = np.linspace(0., 500 , 20)
    
    memory_capacity = Parallel(n_jobs=-1)(delayed(compute_MC_multi)(N= N , k=j, d=i, signal = signal) 
                                          for i in tqdm(d_val) #righe 
                                          for j in k_val) #colonne
    memory_capacity = np.array(memory_capacity).reshape( len(d_val), len(k_val), 10)
    
    
    plt.figure(figsize=(20,10))
    
    
    for i in range(0,10):
        
        z = memory_capacity[:,:,i]
        

        plt.subplot(2, 5, i+1)
        
        plt.pcolor( k_val, d_val ,z, vmin = 0., vmax = 1. )
        plt.xlabel("k")
        plt.ylabel("d")
        plt.title("MC")
        plt.colorbar()
        
    plt.show()


