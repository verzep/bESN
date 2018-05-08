from copy import deepcopy, copy
from sklearn import linear_model
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

import functions_bESN as fb


class bESN:

    # Constructor
    def __init__(self, N, k, d, c=0.5, W_cont=False, x_cont=False, noise = 0):

        self.N = N  # number of neurons [positive integer]
        self.k = k  # asymmetry in the weight distribution [rean number, -0.5 < d < 0.5]
        self.d = d  # mean degree of the network [real number, 0. < k < N]
        self.noise = noise # the noise intensity
        self.c = c
        self.W_cont = W_cont
        self.x_cont = x_cont
        self.x = np.zeros(N)  # current state of the network (each x can be +1 or -1)


        self.W = np.zeros([N, N])  # connectivity matrix (each w can be +1,0 or-1)
        self.W_in = np.ones(N)  # controls which neuron can read the signal (1 if it can, 0 if it can not)
        self.W_fb = np.ones(N)

        self.memory = []  # list of all the past state of the network
        self.energy = []  # list of all the average of the states (REDUNDANT?)
        self.activity = []  # list of all the activity (number of neurons that changed their states) (REDUNDANT?)

        self.every_feedback = []


        if self.x_cont == False:
            self.random_start(c)
        if self.x_cont == True:
            self.random_start_cont(c)
        if self.W_cont == False:
            self.create_W()

        if self.W_cont == True:
            self.create_W_cont()

    # creates W, the weight matrix
    def create_W(self):
        for w in np.nditer(self.W, op_flags=['readwrite']):
            # if the random number is > p := k/N the link is created
            if np.random.rand() < (float(self.k) / (self.N)):
                # if the random number is < a := 1/2 + d the weight is positive, negative otherwise (0 is the default)
                if np.random.rand() < 0.5 + self.d:
                    w[...] = 1
                else:
                    w[...] = -1

    def create_W_cont(self):
        self.random_state_ = np.random.RandomState(42)

        W = self.random_state_.rand(self.N, self.N) - 0.5
        W[self.random_state_.rand(*W.shape) < 0.95] = 0

        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (1.5 / radius)

        self.W_in = self.random_state_.rand(self.N) * 2 - 1

        self.W_fb = self.random_state_.rand(self.N) * 2 - 1

    # compute the spectral radius of W
    def spectral_radius(self):
        e_val, e_vec = LA.eig(self.W)

        return max(np.absolute(e_val))

    # split the neurons in two communities, according to their k_in
    def find_communities(self, threshold):

        k_in = np.count_nonzero(self.W, axis = 1)

        under_k = np.where(k_in<=threshold)
        over_k = np.where(k_in>threshold)

        return under_k[0], over_k[0]



    # randomly generate the starting state (c controls the amount of positive)
    def random_start(self, c=0.5):
        for i in np.nditer(self.x, op_flags=['readwrite']):
            if np.random.rand() < c:
                i[...] = 1
            else:
                i[...] = -1

        self.memory.append(self.x)
        self.energy.append(np.mean(self.x))

    def random_start_cont(self, c=0.5):
        for i in np.nditer(self.x, op_flags=['readwrite']):
            i[...] = np.random.normal(c - 0.5) < c

        self.memory.append(self.x)
        self.energy.append(np.mean(self.x))

    def entropy(self, indices = None):
        if indices is None:
            return [fb.H_b(x) for x in self.memory]
        else:
            return [fb.H_b(x) for x in self.memory[:, indices]]

    def energy(self, indices = None):
        if indices is None:
            return [np.mean(x) for x in self.memory]
        else:
            return [np.mean(x) for x in self.memory [:, indices]]

    def activity(self, indices = None):
        if indices is None:
            return [float(fb.ham_dis(t_1, t_2))/self.N for t_1, t_2 in zip(self.memory[1:], self.memory[:-1])]
        else:
            return [float(fb.ham_dis(t_1, t_2))/self.N for t_1, t_2 in zip(self.memory[1:,indices], self.memory[:-1,indices])]

    def _clean(self):
        start = copy(self.memory[0])

        self.memory = []
        self.energy = []

        self.memory.append(start)
        self.energy.append(np.mean(start))

    def _switch(self, i):
        self.x[i] *= -1

    def _step(self, u=0, y=0, cont=False):
        # x_new = copy(self.x)

        # for i in range(self.N):
        #    h = np.dot(self.W[i],self.x) + u*self.read[i]
        #    
        #    if h:
        #        x_new[i] = np.sign(h)

        self.every_feedback.append(y)

        if cont:
            S = np.dot(self.W, self.x) + np.dot(self.W_in, u) + np.dot(self.W_fb, y) + self.noise * self.k * np.random.normal()
            # x_new = np.tanh( self.W.dot(self.x) + u * self.W_in + y *self.W_fb )

            x_new = np.tanh(S) + 0.001 * (np.random.rand(self.N) - 0.5)
            # x_new[x_new == 0] = self.x[x_new == 0]

        else:
            E = self.noise * self.k * np.random.randn(self.N)
            S = self.W.dot(self.x) + u * self.W_in + y * self.W_fb + E
            # print self.noise
            # print self.k
            # print E[:10]
            # print self.W[0, :10]
            # print u
            # print y






            # th = np.sqrt(self.k) * (1 - self.d)
            x_new = np.sign(S)  # * np.heaviside( np.abs(S)  , 1. )
            x_new[x_new == 0] = self.x[x_new == 0]

            #print S[:10] - E[:10]
            #print E[:10]
            #print x_new[:10]
            #print

        # print(y)

        self.memory.append(x_new)
        self.energy.append(np.mean(x_new))
        self.activity.append(float(fb.ham_dis(x_new, self.x)) / self.N)

        self.x = x_new

    def evolve(self, signal=None, teacher=None, nSteps=500, feedback=False, cont=False):

        if signal is None:
            for i in range(nSteps):
                self._step(cont=cont)
        elif teacher is None and feedback == False:

            for i in range(len(signal)):
                self._step(u=signal[i], cont=cont)

        elif feedback == False:
            print("TEACHER  FORCING!!!!!!!!!!!!")
            for i in range(len(signal)):
                self._step(u=signal[i], y=teacher[i], cont=cont)

        elif feedback == True:
            print("FEEDBACK!!!!!!!!!!!!")

            for i in range(len(signal)):
                x_e = np.hstack((self.x, signal[i]))
                fb = np.dot(self.W_out, x_e)
                # print fb
                self._step(u=signal[i], y=fb, cont=cont)

            # if i%100 == 0 :
            #    print (i, self.energy[i])

        # MM = np.array(self.memory)
        # print "sono MM "
        # print MM[4]
        # plt.imshow(MM.T, aspect='auto', interpolation='nearest')
        # plt.colorbar()

    def perturb(self, i):

        pert = bESN(N =self.N, k = self.k, d = self.d, c= self.c , W_cont=self.W_cont, x_cont=self.x_cont, noise = self.noise)

        pert.W = copy(self.W)
        pert.x = copy(self.memory[0])
        pert._switch(i)

        pert.memory = []
        pert.energy = []

        pert.memory.append(copy(pert.x))
        pert.energy.append(np.mean(pert.x))

        # print ("New", pert.memory)

        return pert

    # Learning

    def fit(self, y, T, settling, signal=0):
        aa = np.array(self.memory[settling + 1:])
        lun = signal[settling: settling + len(self.memory[settling + 1:])]
        ip = np.array(lun).reshape(aa.shape[0], 1)

        print aa.shape, ip.shape
        ext_state = np.hstack((aa, ip))

        self.W_out = np.dot(np.linalg.pinv(ext_state),
                            np.array(y)
                            ).T

        self.ReadOut = linear_model.Ridge(alpha=0.1)
        X_true = self.memory[settling + 1:  1 + T]

        self.ReadOut.fit(X_true, y)

    def predict(self, X):
        return self.ReadOut.predict(X)

    def learn_signal(
            self,
            S,
            delay=0,
            test_length=300,
            make_plot=False,
            settling=300
    ):

        T = len(S) - test_length

        signal_test_evaluate = S[T - delay: T + test_length - delay]

        self.fit(signal_train_evaluate, T=T, settling=settling)

        prev = self.predict(self.memory[T + 1:])

        self.MC = fb.MC(signal_test_evaluate, prev)

        if make_plot:
            train_prev = self.predict(self.memory[settling + 1: 1 + T])

            print max(train_prev), min(train_prev)
            print 'error on the train set ', np.mean((train_prev - signal_train_evaluate) ** 2)

            plt.plot(signal_test_evaluate, label="true")
            plt.plot(prev, label="predicted")
            plt.legend()
            plt.show()

        return np.mean((prev - signal_test_evaluate) ** 2)

    # plot(A.energy[-100:])
    # show()

    def predict_signal(
            self,
            S,
            forecasting=0,
            test_length=300,
            make_plot=False,
            settling=300
    ):

        T = len(S) - test_length - forecasting

        signal_train_evaluate = S[settling + forecasting: T + forecasting]
        signal_test_evaluate = S[T + forecasting: T + test_length + forecasting]

        self.fit(signal_train_evaluate, T=T, settling=settling)

        prev = self.predict(self.memory[T + 1:])

        self.MC = fb.MC(signal_test_evaluate, prev)

        if make_plot:
            train_prev = self.predict(self.memory[settling + 1: 1 + T])

            print 'error on the train set ', np.mean((train_prev - signal_train_evaluate) ** 2)

            plt.plot(signal_test_evaluate, label="true")
            plt.plot(prev, label="predicted")
            plt.legend()
            plt.show()

        return np.mean((prev - signal_test_evaluate) ** 2)
        # plot(A.energy[-100:])
        # show()
