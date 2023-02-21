from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup

from cc3d.core.PySteppables import *

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class CTRNN():

    def __init__(self, size):
        self.Size = size                        # number of neurons in the circuit
        self.States = np.zeros(size)            # state of the neurons
        self.TimeConstants = np.ones(size)      # time-constant for each neuron
        self.invTimeConstants = 1.0/self.TimeConstants
        self.Biases = np.zeros(size)            # bias for each neuron
        self.Weights = np.zeros((size,size))    # connection weight for each pair of neurons
        self.Outputs = np.zeros(size)           # neuron outputs
        self.Inputs = np.zeros(size)            # external input to each neuron

    def setWeights(self, weights):
        self.Weights = weights

    def setBiases(self, biases):
        self.Biases = biases

    def setTimeConstants(self, timeconstants):
        self.TimeConstants = timeconstants
        self.invTimeConstants = 1.0/self.TimeConstants

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(self.Size,self.Size))
        self.Biases = np.random.uniform(-10,10,size=(self.Size))
        self.TimeConstants = np.random.uniform(0.1,5.0,size=(self.Size))
        self.invTimeConstants = 1.0/self.TimeConstants

    def initializeState(self, s):
        self.States = s
        self.Outputs = sigmoid(self.States+self.Biases)

    def step(self, dt):
        netinput = self.Inputs + np.dot(self.Weights.T, self.Outputs)
        self.States += dt * (self.invTimeConstants*(-self.States+netinput))
        self.Outputs = sigmoid(self.States+self.Biases)

    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights, biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.TimeConstants


#Global params for CTRNNs 
size = 3
duration = 100000000
stepsize = 1
time = np.arange(0.0,duration,stepsize)

class testSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)
        #self.track_cell_level_scalar_attribute(field_name='Outputs', attribute_name='Outputs')
        #self.track_cell_level_scalar_attribute(field_name='Node1_activity', attribute_name='Node1')
   
    def start(self):
        """
        any code in the start function runs before MCS=0
        """
#initialize neural network        
        
        for cell in self.cell_list:
            size = 3
            duration = 100
            stepsize = 1
            
            time = np.arange(0.0,duration,stepsize)
            
            cell.dict["CTRNN"] = CTRNN(size)
            cell.dict["CTRNN"].randomizeParameters()
            cell.dict["CTRNN"].initializeState(np.zeros(size))
            
          
        # self.plot_win = self.add_new_plot_window(title='Neural Activity',
                                                     # x_axis_title='MonteCarlo Step (MCS)',
                                                     # y_axis_title='Variables', x_scale_type='linear', y_scale_type='linear',
                                                     # grid=False)
            
        # self.plot_win.add_plot("activity", style='Lines', color='red', size=5)
            
   
    def step(self,mcs):
        """
        type here the code that will run every frequency MCS
        :param mcs: current Monte Carlo step
        """
        for cell in self.cell_list:
            cell.dict["CTRNN"].step(stepsize)
            outputs[mcs] = cell.dict["CTRNN"].Outputs
            
            # arguments are (name of the data series, x, y)
            # self.plot_win.add_data_point("activity", mcs, outputs)
        
    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return


        