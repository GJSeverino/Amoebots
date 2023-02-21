from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup
import numpy as np
from cc3d.core.PySteppables import *
import random as rd
from statistics import mean
from array import array
import os

a = 0.9
b = 0.0
#rs = 1

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
        
class UniCell_Explicit_Force_2DSteppable(SteppableBasePy):

    
    def add_steering_panel(self):
        self.add_steering_param(name='Force Persistence', val=0.9, min_val=0.0, max_val=1.0,
                                decimal_precision=3, widget_name='slider')
        self.add_steering_param(name='Force Directional Noise', val=0.00, min_val=0.0, max_val=1.0,
                                decimal_precision=2, widget_name='slider')
        self.add_steering_param(name='Surface Tension', val=-5., min_val=-5.0, max_val=5.0,
                                decimal_precision=1, widget_name='slider')
        self.add_steering_param(name='Number of Cells', val=4, min_val=1, max_val=200,
                                decimal_precision=0, widget_name='slider')
        self.add_steering_param(name='Cell Volume', val=50, enum=[30,40,50,60,70,80,90,100],
                                    widget_name='combobox')
        self.add_steering_param(name='Lambda Volume', val=1., min_val=0.0, max_val=100.0,
                                decimal_precision=1, widget_name='slider')
    
    def process_steering_panel_data(self):
        print ('processing steering panel updates')
        print ('all dirty flag=', self.steering_param_dirty())
        self.FP = self.get_steering_param('Force Persistence')
        self.FDN = self.get_steering_param('Force Directional Noise')
        self.ST = self.get_steering_param('Surface Tension')
        self.NC = self.get_steering_param('Number of Cells')
        self.CV = self.get_steering_param('Cell Volume')
        self.LV = self.get_steering_param('Lambda Volume')
        print ('updated MY_PARAM_SLIDER=',self.FP)
        print ('updated MY_PARAM_COMBO=', self.CV)
        Jcc = self.get_xml_element('Jcc')
        Jmc = self.get_xml_element('Jmc')
        print(dir(Jcc))
        Jcc.cdata = self.ST + 2.*float(Jmc.cdata)
        # IMPORTANT: you may add code here tht updates cell properties based on the
        # values of the parameters from the steering panel. For example:
        for cell in self.cell_list:
            cell.targetVolume = self.CV
            cell.lambdaVolume = self.LV
        self.alpha = self.FP
        self.beta = self.FDN
        
        N0 = len(self.cell_list)
        
        if self.NC>N0 :
            while self.NC!=len(self.cell_list):
                while True:
                    
                    x = np.random.uniform(3, self.dim.x - 3)
                    y = np.random.uniform(3, self.dim.y - 3)
                
                    if not self.cell_field[x, y, 0]:
                        break
                
                cell = self.new_cell(self.UNICELL)
                
                cell.targetVolume = self.CV
                cell.lambdaVolume = self.LV
                self.cell_field[(x-1):(x+1), (y-1):(y+1), 0] = cell
                
                theta = np.random.uniform(0,2*np.pi)
                Fx = np.cos(theta)
                Fy = np.sin(theta)
                Fz = 0
                cell.dict["Old_pos"] = [cell.xCOM,cell.yCOM,cell.zCOM]
                cell.dict["ExForce"] = [Fx, Fy, Fz]
                cell.dict["Scale"] = 100.0
                cell.dict["Theta"] = theta
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX = cell.dict["Scale"]*cell.dict["ExForce"][0]
                cell.lambdaVecY = cell.dict["Scale"]*cell.dict["ExForce"][1]
                cell.lambdaVecZ = cell.dict["Scale"]*cell.dict["ExForce"][2]
            
                
        elif self.NC<N0 :
            while self.NC!=len(self.cell_list):
                
                for cell in self.cell_list:
                    self.delete_cell(cell)
                    break
                
                
    
    def __init__(self, frequency=1):

        self.create_vector_field_cell_level_py("Polarization")
        
        
        SteppableBasePy.__init__(self,frequency)
        self.alpha = a
        self.beta = b
        self.CV = 50
        self.LV = 1.
        #self.memory = 0.4 #must be less than 0.5
        
            
    def start(self):
        
        """
        Called before MCS=0 while building the initial simulation
        """
        #self.plot_win = self.add_new_plot_window(title='Theta',
                                                 #x_axis_title='MonteCarlo Step (MCS)',
                                                 #y_axis_title='Theta', x_scale_type='linear', y_scale_type='linear',
                                                 #grid=False)
        
        
      
        
        #self.plot_win.add_plot("Theta", style='Dots', color='red', size=5)
        #theta0 = np.pi/4 + np.pi
        
        for cell in self.cell_list:
            
            #Initializing neural network 
            
            size = 3 
            duration = 100  # mcs? 
            stepsize = 0.1
            time = np.arange(0.0,duration,stepsize)
            
            
            cell.dict["CTRNN"] = CTRNN(size)
            cell.dict["CTRNN"].randomizeParameters()
            cell.dict["CTRNN"].initializeState(np.zeros(size))
            
            outputs = np.zeros((len(time),size))
            
            # 5 - 17 nn /. cell.dict 
            cell.targetVolume = self.CV
            cell.lambdaVolume = self.LV
            theta = np.random.uniform(0,2*np.pi)
            Fx = np.cos(theta)
            Fy = np.sin(theta)
            Fz = 0
            cell.dict["Old_pos"] = [cell.xCOM,cell.yCOM,cell.zCOM]
            cell.dict["ExForce"] = [Fx, Fy, Fz]
            cell.dict["Scale"] = 100.0
            cell.dict["Theta"] = theta
            
            # Make sure ExternalPotential plugin is loaded
            cell.lambdaVecX = cell.dict["Scale"]*cell.dict["ExForce"][0]
            cell.lambdaVecY = cell.dict["Scale"]*cell.dict["ExForce"][1]
            cell.lambdaVecZ = cell.dict["Scale"]*cell.dict["ExForce"][2]
            
            ##plotting
            
            self.plot_win = self.add_new_plot_window(title='Output from CTRNN',
                                                     x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Output', x_scale_type='linear', y_scale_type='linear',
                                                     grid=False)
            
            self.plot_win.add_plot("nnoutputs", style='Lines', color='red', size=5)
            
            
            

    def step(self, mcs):
        """
        Called every frequency MCS while executing the simulation
        
        :param mcs: current Monte Carlo step
        """
        
        for cell in self.cell_list:
            # 22 - 23 
            cell.dict["CTRNN"].step(stepsize)
            outputs[step] == cell.dict["CTRNN"].Outputs
        
        
        ## This is for plotting 
            # arguments are (name of the data series, x, y)
        
        self.plot_win.add_data_point("nnoutputs", mcs, outputs)
            
            
            
        
        
        if mcs%10 == 0:
            
            
            field = self.field.Polarization
            field.clear()
            for cell in self.cell_list:
                
                theta = np.random.vonmises(0.,4.)
                Fx_noise = np.cos(theta)
                Fy_noise = np.sin(theta)
                
                Current_pos = [cell.xCOM,cell.yCOM,cell.zCOM]
                
                Vx = Current_pos[0] - cell.dict["Old_pos"][0]
                Vy = Current_pos[1] - cell.dict["Old_pos"][1]
                Vz = Current_pos[2] - cell.dict["Old_pos"][2]
                
                if Vx > self.dim.x/2.:
                    Vx -= self.dim.x
                if Vx < -self.dim.x/2.:
                    Vx += self.dim.x
                if Vy > self.dim.y/2.:
                    Vy -= self.dim.y
                if Vy < -self.dim.y/2.:
                    Vy += self.dim.y
                
                Norm = np.sqrt(Vx*Vx+Vy*Vy+Vz*Vz)
                
                if Norm > 0:
                
                    theta_v = np.arctan2(Vy,Vx)
                    theta_f = np.arctan2(cell.lambdaVecY,cell.lambdaVecX)
                    
                    
                    dif = theta_f - theta_v
                    if dif > np.pi: dif = -2*np.pi + dif
                    if dif < -np.pi: dif = 2*np.pi + dif
                    
                    theta_f += (1-self.alpha)*dif
                    if theta_f > np.pi: theta_f = -2*np.pi + theta_f
                    if theta_f < -np.pi: theta_f = 2*np.pi + theta_f
                    
                    theta_f += theta*self.beta
                    if theta_f > np.pi: theta_f = -2*np.pi + theta_f
                    if theta_f < -np.pi: theta_f = 2*np.pi + theta_f

                    
                    cell.dict["Theta"] = theta_f
                    
                    Fx = np.cos(theta_f)
                    Fy = np.sin(theta_f)
                    field[cell] = [-2*Fx, -2*Fy, 0]
                    
                    
                    
                    
                    Fz = 0.
                    
                    # Fx = cell.dict["ExForce"][0]*self.alfa + (1-self.alfa)*Vx/Norm
                    # Fy = cell.dict["ExForce"][1]*self.alfa + (1-self.alfa)*Vy/Norm
                    # Fz = cell.dict["ExForce"][2]*self.alfa + (1-self.alfa)*Vz/Norm
                    
                    # Fx = Fx*self.beta + (1-self.beta)*Fx_noise
                    # Fy = Fy*self.beta + (1-self.beta)*Fy_noise
                
                    self.plot_win.add_data_point("Theta", mcs, np.arctan2(Fy,Fx))
                    
                    FNorm = np.sqrt(Fx*Fx+Fy*Fy+Fz*Fz)
                    
                    # cell.dict["ExForce"] = [Fx/FNorm, Fy/FNorm, Fz/FNorm]
                    cell.dict["ExForce"] = [Fx, Fy, Fz]
                    
                    cell.lambdaVecX = cell.dict["Scale"]*cell.dict["ExForce"][0]
                    cell.lambdaVecY = cell.dict["Scale"]*cell.dict["ExForce"][1]
                    # cell.lambdaVecZ = cell.dict["Scale"]*cell.dict["ExForce"][2]
                    
                    #print(cell.dict["Old_pos"][:])
                    cell.dict["Old_pos"][0] = Current_pos[0]
                    cell.dict["Old_pos"][1] = Current_pos[1]
                
                
    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
        
class ClusterCountSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
        
        

    def start(self):
        
        self.plot_win = self.add_new_plot_window(title='Cluster Count',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Number', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)
        
        self.plot_win.add_plot("Count", style='Dots', color='red', size=5)
        # initialize setting for Histogram
        self.plot_win2 = self.add_new_plot_window(title='Histogram of Cluster Sizes', x_axis_title='Cluster Size',
                                                 y_axis_title='Number of Clusters')
        # _alpha is transparency 0 is transparent, 255 is opaque
        self.plot_win2.add_histogram_plot(plot_name='Hist 1', color='green', alpha=100)
        
        
        
    def step(self, mcs):
        
        if mcs%100==0 and mcs>500:
        
            #This first routine basically identifies all neighboring cells for all cells independently of cell type
            #Medium cell is obviously not counted. If you have ECM or cells that do not participate in the cluster formation
            #you have to loop over cells of a given type or simply not count cells of a certain type
            
            #Count the number of cells
            self.N = 0
            for cell in self.cell_list:
                self.N += 1
            #since each cell requires a list of neighbors, a matrix is required
            self.M = [[] for i in range(self.N)]
            i=0
            for cell in self.cell_list:
                self.M[i].append(cell.id)
                #find the neighbors of each cells and assign them to lists, each list is a neighborhood
                for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                    if neighbor:
                        self.M[i].append(neighbor.id)
                    else:
                        pass
                i+=1
            
            #The following routine reduce all lists sharing elements into a single list,
            #while reduced lists not sharing elements remain separate.
            #This do the job of finding the clusters from starting lists of neighborhoods
            
            CONTINUE = 1
            while CONTINUE:
                CONTINUE = 0
                for i in range(self.N):
                    for j in range(self.N):
                        if i!=j:
                            if not set(self.M[i]).isdisjoint(self.M[j]): #if intersection exists between two neighborhoods i and j
                                self.M[i] = self.M[i]+self.M[j] #join lists
                                self.M[i] = list(set(self.M[i])) #remove duplicates
                                self.M[j] = [] #clear j entry to prevent process repetition when process ends
                                CONTINUE = 1 #allow process repetition
            count=0
            size = []
            for i in range(self.N):
                if self.M[i]:
                    count+=1
                    size.append(len(self.M[i]))
            #print(size)
            #print(count)
                    # arguments are (name of the data series, x, y)
            self.plot_win.add_data_point("Count", mcs, count)
            #self.plot_win2.add_histogram(plot_name='Hist 1', value_array=size, number_of_bins=20)
            
            
            
                    
            


    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return
        
class CalculationsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        '''
        constructor
        '''
        SteppableBasePy.__init__(self, frequency)
        # PLACE YOUR CODE BELOW THIS LINE
        

    def start(self):
        self.Ncell=0
        for compartments in self.clusters:
           self.Ncell+=1
        self.file1 = open(r"D:\CompuCell3D-py3-64bit\Simulations\UniCell_Explicit_Force_2D\Output.txt","a")
        self.file1.write("Distance \t p.p \n")

    def step(self, mcs):
        
        #calculate polarity and center of mass of a random cell
        dimx = self.dim.x
        dimy = self.dim.y
        cell_id = rd.randint(1, self.Ncell) #CHOOSE A RANDOM CELL
        _cell = self.fetch_cell_by_id(cell_id)
        _vol = _cell.volume
        _cm = [_cell.xCOM,_cell.yCOM]
        
        _Polarity = [_cell.dict["ExForce"][0],_cell.dict["ExForce"][1]]
        
        #print(_cm)
        
        #print the polarity product and the distance between this cell and all the others
        for cell in self.cell_list:
            cell_vol = cell.volume
            cell_cm = [cell.xCOM,cell.yCOM]
            
            Polarity = [cell.dict["ExForce"][0],cell.dict["ExForce"][1]]
            CM = [cell.xCOM,cell.yCOM]
            
            if CM[0]-_cm[0] > dimx/2.: correctx = -1
            elif CM[0]-_cm[0] < -dimx/2.: correctx = 1
            else: correctx = 0
            if CM[1]-_cm[1] > dimy/2.: correcty = -1
            elif CM[1]-_cm[1] < -dimy/2.: correcty = 1
            else: correcty = 0
            
            CM = np.add(CM,[dimx*correctx,dimy*correcty])
            
            P_product = np.inner(Polarity,_Polarity)
            
            distance = np.linalg.norm(np.subtract(CM,_cm))
            
            if (100>distance>0):
                self.file1.write(str(distance)+"\t"+str(P_product)+"\n")

    def finish(self):
        '''
        this function may be called at the end of simulation - used very infrequently though
        '''        
        # PLACE YOUR CODE BELOW THIS LINE
        self.file1.close()
        return

    def on_stop(self):
        '''
        this gets called each time user stops simulation
        '''        
        # PLACE YOUR CODE BELOW THIS LINE
        
        return
        
class PersistentNeighborsSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):
        
        out_dir_name = "UCEF2D"
        print (out_dir_name)
        if not os.path.exists(out_dir_name): os.makedirs(out_dir_name)
        cell = self.fetch_cell_by_id(2)
        file_name = "PN_a"+str(a)+"_b"+str(b)+"_F"+str(cell.dict["Scale"])#+"_rs"+str(rs)
        self.output_path = str(Path(out_dir_name+"\\"+file_name))
        self.file4 = open (self.output_path, 'w')
        self.file4.write("DeltaT \t PN \n")
        
        self.samples = 100
        self.DTmin = 100
        
        for cell in self.cell_list:
            cell.dict["ListN"] = np.zeros((self.samples,10))
            
        self.count1 = 0
            
    def step(self, mcs):
        waiting_time = 1000
        if mcs > waiting_time :
            if (mcs-waiting_time)%self.DTmin == 0 :
                for cell in self.cell_list:
                    self.count2 = 0
                    for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                        if neighbor:
                            cell.dict["ListN"][self.count1][self.count2] = neighbor.id+1
                            self.count2 += 1
                        else:
                            cell.dict["ListN"][self.count1][self.count2] = 1
                            self.count2 += 1
                self.count1 += 1
                            
            if (mcs-waiting_time)%(self.samples*self.DTmin) == 0 :
                
                for cell in self.cell_list:
                    List0 = cell.dict["ListN"][0][:]
                    List0 = [i for i in List0 if i != 0]
                    for count3 in range(self.samples):
                        List_dt = cell.dict["ListN"][count3][:]
                        dt = self.DTmin*count3
                        #CN_list = [x for x in np.concatenate((List0,List_dt)) if x not in List0 or x not in List_dt]
                        CN_list = [x for x in List0 if x not in List_dt]
                        #CN = len(CN_list)*0.5
                        CN = len(CN_list)/len(List0)
                        #print(List0, List_dt)
                        #print(CN_list)
                        if not 1 in np.concatenate((List0,List_dt)):
                            self.file4.write(str(dt)+"\t"+str(CN)+"\n")
                        #arr = np.ndarray(shape=2,buffer=np.array([dt,CN]))
                        #arr = array("d",[dt,CN])
                        #arr.tofile(self.file4)
                
                self.count1 = 0
                for cell in self.cell_list:
                    cell.dict["ListN"] = np.zeros((self.samples,10))
                    
    def finish(self):
        
        self.file4.close()
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return


        
class CollectivityCalcSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        
        self.Collectivity = []
        self.phi = []
        self.gama = []
        
    def step(self, mcs):
        if mcs > 10000 and mcs%10 == 0 :
            count = 0
            phi_x = 0
            phi_y = 0
            L = 0
            for cell in self.cell_list:
                count += 1
                Fx = np.cos(cell.dict["Theta"])
                Fy = np.sin(cell.dict["Theta"])
                #print (Fx, cell.dict["ExForce"][0])
                X = cell.xCOM
                Y = cell.yCOM
                L += (- Fx*(Y-self.dim.y/2.) + Fy*(X-self.dim.x/2.))/np.sqrt((X-self.dim.x/2.)**2 + (Y-self.dim.y/2.)**2)
                phi_x += Fx
                phi_y += Fy
            phi_x /= count
            phi_y /= count
            P = np.sqrt(phi_x**2 + phi_y**2)
            L = np.sqrt(L**2)/count
            self.phi.append(P)
            self.gama.append(L)
            self.Collectivity.append(np.sqrt(P**2 + L**2))
        
    def finish(self):
        
        out_dir_name = "UCEF2D"
        print (out_dir_name)
        if not os.path.exists(out_dir_name): os.makedirs(out_dir_name)
        file_name = "a_phi_gama_col_a"+str(a)+"_b"+str(b)#+"_rs"+str(rs)
        self.output_path = str(Path(out_dir_name+"\\"+file_name))
        self.file3 = open (self.output_path, 'a')
        self.file3.write(str(a)+"\t"+str(mean(self.phi))+"\t"+str(mean(self.gama))+"\t"+str(mean(self.Collectivity))+"\n")
        #self.file3.write("\t"+str(mean(self.gama))+"\n")
        #self.file3.write("\t"+str(mean(self.Collectivity))+"\n")
        self.file3.close()
        
        return

    def on_stop(self):
        
        return


        
class Position_OutputSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):
        
        out_dir_name = "UCEF2D"
        print (out_dir_name)
        if not os.path.exists(out_dir_name): os.makedirs(out_dir_name)
        file_name = "_a"+str(a)+"_b"+str(b)#+"_rs"+str(rs)
        self.output_path = str(Path(out_dir_name+"\\"+file_name))
                
        self.file4 = open (self.output_path, 'wb') #AAAAAA
        #self.file4.write("MCS \t")#AAAAAA
        for cell in self.cell_list:
            #self.file4.write("X "+str(cell.id)+"\t Y "+str(cell.id)+"\t Px "+str(cell.id)+"\t Py "+str(cell.id)+"\t")#AAAAAA
            cell.dict["cx"] = 0
            cell.dict["cy"] = 0
            cell.dict["Old_pos2"] = [cell.xCOM,cell.yCOM,cell.zCOM]
        #self.file4.write("\n")#AAAAAA

    def step(self, mcs):
        
        if mcs>100:
            #self.file4.write(str(mcs)+"\t")#AAAAAA
            list = []
            for cell in self.cell_list:
                current_pos = [cell.xCOM,cell.yCOM,cell.zCOM]
                if cell.xCOM - cell.dict["Old_pos2"][0] > self.dim.x*0.5: cell.dict["cx"] -= 1
                if cell.xCOM - cell.dict["Old_pos2"][0] < -self.dim.x*0.5: cell.dict["cx"] += 1
                if cell.yCOM - cell.dict["Old_pos2"][1] > self.dim.y*0.5: cell.dict["cy"] -= 1
                if cell.yCOM - cell.dict["Old_pos2"][1] < -self.dim.y*0.5: cell.dict["cy"] += 1
                #self.file4.write(str(cell.xCOM+self.dim.x*cell.dict["cx"])+"\t"+str(cell.yCOM+self.dim.y*cell.dict["cy"])+"\t")#AAAAAA
                #self.file4.write(str(cell.lambdaVecX)+"\t"+str(cell.lambdaVecY)+"\t")#AAAAAA
                cell.dict["Old_pos2"][:] = current_pos[:]
                list.append(cell.xCOM+self.dim.x*cell.dict["cx"])#AAAAAA
                list.append(cell.yCOM+self.dim.y*cell.dict["cy"])#AAAAAA
                list.append(cell.lambdaVecX)#AAAAAA
                list.append(cell.lambdaVecY)#AAAAAA
            
            arr = array("d",list)#AAAAAA
            arr.tofile(self.file4)#AAAAAA
            
            #self.file4.write("\n")#AAAAAA

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        self.file4.close()
        return

        
class Interface_With_MediumSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):
        
        out_dir_name = "UCEF2D"
        print (out_dir_name)
        if not os.path.exists(out_dir_name): os.makedirs(out_dir_name)
        file_name = "Interface_a"+str(a)+"_b"+str(b)#+"_rs"+str(rs)
        self.output_path = str(Path(out_dir_name+"\\"+file_name))
        self.file4 = open (self.output_path, 'w') 
        self.Interfaces = []


    def step(self, mcs):
        
        if mcs>1000 and mcs%100==0:
            total_area = 0
            for cell in self.cell_list:
                for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                    if neighbor:
                        pass
                    else:
                        total_area += common_surface_area
            self.Interfaces.append(total_area)


    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        avg_Interface = sum(self.Interfaces)/len(self.Interfaces)
        self.file4.write("Average Interface With Medium \n" + str(avg_Interface))
        self.file4.close()
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return


        



