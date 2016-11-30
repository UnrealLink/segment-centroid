
import numpy as np
import copy
from sklearn.preprocessing import normalize
from scipy.misc import logsumexp

class ForwardBackward(object):
    """
    The ForwardBackward class performs one forward and backward pass
    of the algorithm returning the tables Q[t,h], B[t,h]. These define
    the weighting functions for the gradient step. The class is initialized
    with model and logging parameters, and is fit with a list of trajectories.
    """

    def __init__(self, model, verbose=True):
        """
        model: This is a model object which is a wrapper for a tensorflow model
        verbose: True means that FB algorithm will print logging output to stdout
        """

        self.verbose = verbose

        if self.verbose:
            print("[HC: Forward-Backward] model=",model)

        self.model = model
        self.k = model.k

        self.X = None
        self.Q = None
        self.B = None
        self.T = None

        self.P = np.ones((self.k, self.k))/self.k


    def fit(self, trajectoryList):
        """
        Each trajectory is a sequence of tuple (s,a) where both s and a are numpy
        arrays. 
        trajectoryList: is a list of trajectories.

        Returns a dict of trajectory id and the tables
        """

        iter_state = {}

        if self.verbose:
            print("[HC: Forward-Backward] found ",len(trajectoryList),"trajectories")

        for i, traj in enumerate(trajectoryList):

            if self.verbose:
                print("[HC: Forward-Backward] fitting ", i, len(traj))

            self.init_iter(i, traj)
            
            iter_state[i] = self.fitTraj(traj)

        return iter_state


    def init_iter(self,index, X):
        """
        Internal method that initializes the state variables
        """

        self.Q = np.ones((len(X)+1, self.k))/self.k
        self.fq = np.ones((len(X)+1, self.k))/self.k
        self.bq = np.ones((len(X)+1, self.k))/self.k

        #uniform probability of next timestep terminating
        self.B = np.ones((len(X)+1, self.k))/2

        self.T = np.zeros((len(X), self.k))

        #uniform transition probabilities
        self.P = np.ones((self.k, self.k))/self.k



    def fitTraj(self, X):
        """
        This function runs one pass of the Forward Backward algorithm over
        a trajectory.
        X is a list of s,a tuples.
        Returns two tables Q, B which define the weights for the gradient step
        """

        self.X = X

        self.forward()
        self.backward()

        self.Q = np.exp(np.add(self.fq,self.bq))
        self.Q = normalize(self.Q, norm='l1', axis=1)

        if self.verbose:
            print("[HC: Forward-Backward] Q Update", np.argmax(self.Q, axis=1))            

        for t in range(len(self.X)-1):
            self.B[t,:] = np.exp(self.termination(t))

        if self.verbose:
            print("[HC: Forward-Backward] B Update", np.argmax(self.B, axis=1)) 


        for t in range(len(self.X)):
            for i in range(self.k):
                self.T[t,i] = self._pi_term_giv_s(X[t][0],i)

        return self.Q[0:len(X),:], self.B[0:len(X),:], self.T


    def forward(self):
        """
        Performs a foward pass, updates the state
        """

        t = len(self.X)-1

        #initialize table
        forward_dict = {}
        for h in range(self.k):
            forward_dict[(0,h)] = 0

        #dynamic program
        for cur_time in range(t):

            state = self.X[cur_time][0]
            next_state = self.X[cur_time+1][0]
            action = self.X[cur_time][1]

            for hp in range(self.k):

                forward_dict[(cur_time+1, hp)] = \
                            logsumexp([ forward_dict[(cur_time, h)] + \
                                        np.log(self._pi_a_giv_s(state,action,h)) + \
                                        np.log(self.P[hp,h]) + 
                                        np.log(self._pi_term_giv_s(next_state,h) + \
                                              (1-self._pi_term_giv_s(next_state,h))*(hp == h))\
                                        for h in range(self.k)
                                     ])

                if self.verbose:
                    print("[HC: Forward-Backward] Forward DP Update", forward_dict[(cur_time+1, hp)], hp, cur_time+1) 

        for k in forward_dict:
            self.fq[k[0],k[1]] = forward_dict[k]


    def backward(self):
        """
        Performs a backward pass, updates the state
        """

        t = 0

        #initialize table
        backward_dict = {}
        for h in range(self.k):
            backward_dict[(len(self.X),h)] = 0

        rt = np.arange(len(self.X), t, -1)

        #dynamic program
        for cur_time in rt:

            state = self.X[cur_time-1][0]
            
            if cur_time == len(self.X):
                next_state = state
            else:
                next_state = self.X[cur_time][0]

            action = self.X[cur_time-1][1]

            for h in range(self.k):
                
                backward_dict[(cur_time-1, h)] = \
                    logsumexp([ backward_dict[(cur_time, hp)] +\
                                np.log(self._pi_a_giv_s(state,action,h)) +\
                                np.log((self.P[hp,h]*self._pi_term_giv_s(next_state,h) + \
                                                                 (1-self._pi_term_giv_s(state,h))*(hp == h)
                                                                ))\
                                for hp in range(self.k)
                              ])

                if self.verbose:
                    print("[HC: Forward-Backward] Backward DP Update", backward_dict[(cur_time-1, h)], h, cur_time-1) 

        for k in backward_dict:
            self.bq[k[0],k[1]] = backward_dict[k]


    def termination(self, t):
        """
        This function calculates B for a particular time step
        """

        state = self.X[t][0]
            
        if t+1 == len(self.X):
            next_state = state
        else:
            next_state = self.X[t+1][0]

        action = self.X[t][1]


        termination = {}

        for h in range(self.k):

            termination[h] = \
                logsumexp([self.fq[t,h] + \
                           np.log(self._pi_a_giv_s(state,action, h)) + \
                           np.log(self.P[hp,h]) + \
                           np.log(self._pi_term_giv_s(next_state, h)) + \
                           self.bq[t+1,hp] \
                           for hp in range(self.k)
                          ])

            if self.verbose:
                    print("[HC: Forward-Backward] Termination Update", termination[h], h) 


        return [termination[h] for h in range(self.k)]


    def _pi_a_giv_s(self, s, a, index):
        """
        Wrapper for the model function
        """
        return self.model.evalpi(index, s, a)


    def _pi_term_giv_s(self, s, index):
        """
        Wrapper for the model function
        """
        return self.model.evalpsi(index, s)




        