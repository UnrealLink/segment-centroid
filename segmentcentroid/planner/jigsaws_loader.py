"""
This class defines a loader that loads 
the JHU Jigsaws dataset
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from .AbstractPlanner import *

class JigsawsPlanner(AbstractPlanner):

    def __init__(self, directory, arms=['left', 'right']):

        #declaring variables that should be set
        self.directory = directory
        self.arms = arms

        if arms == ['left', 'right']:
            self.START = 39
            self.END = 76

        elif 'left' in arms:
            self.START = 39
            self.END = 58

        elif 'right' in arms:
            self.START = 58
            self.END = 76

        else:

            raise ValueError("Invalid Arm Config")


        super(JigsawsPlanner, self).__init__(None)


    """
    This function returns a trajectory [(s,a) tuples]
    """
    def plan(self, max_depth=-1, start=None):
        files = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
        index = np.random.choice(len(files))
        f = open(self.directory+"/"+files[index], "r")
        
        lines = f.readlines()
        states = [np.array([float(li) for li in l.split()][self.START:self.END]) for l in lines]

        if max_depth == -1:
            max_depth = len(states)
        else:
            max_depth = min(max_depth, len(states))

        traj = []

        for t in range(1, max_depth):
            xt = states[t-1]
            xtp = states[t]
            traj.append((xt, xtp-xt))

        return traj


    def visualizePlans(self, trajs, model, filename=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        colors = np.random.rand(model.k,3)

        for i, traj in enumerate(trajs):
            Q = np.argmax(model.fb.fit([traj])[0][0], axis=1)
            segset = { j : np.where(Q == j)[0] for j in range(model.k) }

            for s in segset:

                plt.scatter(segset[s], segset[s]*0 + i, color=colors[s,:])

        if filename == None:
            plt.show()
        else:
            plt.savefig(filename)
            




        
