'''
Original Chemokinesis simulation by Pedro Cenci Dal Castel, contact: pdalcastel@gmail.com 
04/22/2023, Brazil

This simulation models chemokinesis, a biological cellular phenomenon when cells perform direct
migration according to field change in time rather than field change in space.

It is observed that, in chemokinesis, cells have intercalated periods of persistent migration and
tumbling behavior (reorientation). If the field change in time is positive, cells tumble less and
persist more. If the field change in time is negative, cells tumble more and persist less.

To regulate cell behavior, there are 3 important parameters: 

1) Persistence
2) Force
3) Sensitivity

And a few mechanisms:

1) Cell polarization reorientation during tumbling
    - depends on Persistence
2) Cell external potential acting over cell's center of mass in the direction of polarization
    - depends on Force and polarization direction
3) Modulation of persistence by the field change in time (negative field change in time means less
persistence, while positive field change in time means more persistence)
    - depends on fiend change in time and sensitivity
4) Modulation of probability to tumble by the field change in time (positive field change in time 
means lower probability to tumble, while positive field change in time means greater probability 
to tumble)
    - depends on fiend change in time and sensitivity
    
    
05/2/23:    
    
Deep learning for a Chemokinesis model: a proof of principle study. 

Gabriel J. Severino and Joel Vanin, IU Bloomington

gjseveri@iu.edu

Using the chemokinesis code provided and explained above, we provide an example study to show how a neural network might be used within CC3D 
to search for optimal parameters. This study serves as a proof of principle that tensorflow can be integrated within CC3D in a minimal, convienent, 
and effeicient manner. 


The job of the neural network will be to find the parameters that find the most optimal angle in which to approach the food. 

The outputs of the neural network will be: 

PT: the probability that the cell will tumble. 
Persistance: how much the cell will change direction each time it tumbles
Sensetivity: sensetivity to the field change in time; translates into effectice persistance and probability to tumble

The input of the neural network will be the current concentration and a limited memory of the history of the concentration. 
'''

from cc3d.core.PySteppables import *
import random as rd    
import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers 
import os 


class ChemokinesisSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency) 
        
       # RUN AND TUMBLE PARAMETERS (begin as random to help search process) 
        
        self.Persistence =  rd.randint(0,200) #100.0                             #cell persistence (inverse of the noise, or inverse of the von mises distribution width)
        self.Force =  200.0                                                      #this will be the external force lambda parameter
        self.Sensitivity = rd.randint(0,200) #100.0                              #sensitivity to the field change in time
        self.PT = rd.uniform(0,1)                                                #probability to tumble (base value). The REAL PT value is actually "cell.dict["PT"]"
        
        # MODEL DEFINITIONS 
        
        self.load_existing_model = False
        self.model = self.load_or_train_model() # Create tensorflow model
        self.memory = [] # Memory to store experiences (state, action, reward, next_state)
        self.network_timestep = 20
        self.gamma = 0.95 # Discount factor for Q-learning
        self.add_steering_param(name='load_existing_model',val='True' if self.load_existing_model else 'False',enum=['True','False'], widget_name='combobox')

        '''
        Persistence is related to how much the cell will change direction each time it tumbles.
        Try persistence {1., 100., 10000.}
        
        Force is related to the cell drift speed in the direction of polarization. 
        Try force = {100, 200, 1000}
        
        Sensitivity is how much field change translates into effective persistence and effective probability to tumble.
        Try sensitibity = {0., 100., 1000.} to see how the chemokinetic response changes.
        
        Remember that force, sensitivity and persistence are codependent in chemokinesis. You should scan all three together.
        '''
############## CREATING TENSORFLOW MODEL ################################

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def start(self):
        
################# INITIALIZING RUN AND TUMBLE PARAMS AND MODEL ########################

        self.model = self.create_model()
        self.replay_buffer = []
        F = self.field.Field
        
        for cell in self.cell_list:
            cell.dict["P"] = rd.uniform(-np.pi,np.pi)            #initialize a random polarization direction
            cell.dict["F_now"] = F[cell.xCOM, cell.yCOM, 0]      #initialize field value
            cell.dict["F_old"] = cell.dict["F_now"]              #initialize old field value
            cell.dict["Persistence"] = self.Persistence          #initialize persistence
            cell.dict["PT"] = self.PT                            #Initialize PT, where PT = probability to tumble
            cell.dict["Field_History"] = [cell.dict["F_now"]]*10 #this is where the cell stores the history of field values
            
            '''
            I chose 10 to be the memory length of the cell to lower the effect of potts noise in the field change information
            as a function of cell displacement. You can choose other values. Whatever number you use here, don't forget to cut 
            the first same amount of numbers out of the data output.
            '''
            
            #FORCE
            cell.lambdaVecX = -self.Force*np.cos(cell.dict["P"]) #initialize force x coordinate (has to be negative)
            cell.lambdaVecY = -self.Force*np.sin(cell.dict["P"]) #initialize force y coordinate
            cell.lambdaVecZ = 0.0                                #initialize force z coordinate (zero because simulation is 2D)
           
          
         
        
       
################################ INTIALIZING PLOTTING WINDOWS #######################################
            self.plot_win = self.add_new_plot_window(title='Cell Position',
                                                     x_axis_title='x position',
                                                     y_axis_title='y position', x_scale_type='linear', y_scale_type='linear',
                                                     grid=False)
            
            
            self.plot_win.add_plot("POS", style='Dots', color='green', size=1)
            
             # Create a new plot window for reward over time
            self.reward_plot = self.add_new_plot_window(title='Reward over Time',
                                                x_axis_title='MCS',
                                                y_axis_title='Reward',
                                                x_scale_type='linear',
                                                y_scale_type='linear',
                                                grid=True)

            # Add a plot to the window
            self.reward_plot.add_plot("Reward", style='Dots', color='blue', size=1)
            
            # Create a new plot window for loss over time
            self.loss_plot = self.add_new_plot_window(title='Loss over Time',
                                              x_axis_title='MCS',
                                              y_axis_title='Loss',
                                              x_scale_type='linear',
                                              y_scale_type='linear',
                                              grid=True)

            # Add a plot to the window
            self.loss_plot.add_plot("Loss", style='Dots', color='red', size=1)
        
    def step(self,mcs):

        F = self.field.Field #invoke field object
        gamma = 0.9 #discount factor
        batch_size = 32 #batch size for training
        update_frequency = 100 #update frequency for training

        for cell in self.cell_list:

###################CHEMOKINESIS (RUN AND TUMBLE) DEFINITIONS ###########################
            cell.dict["F_now"] = F[cell.xCOM, cell.yCOM, 0]             #get current field value at cell center of mass
            cell.dict["Field_History"][mcs%10] = cell.dict["F_now"]
            cell.dict["F_old"] = cell.dict["Field_History"][(mcs+1)%10] #take the field value 10 steps before current step
            F_change = cell.dict["F_now"] - cell.dict["F_old"]          #calculates the field change during the 10 previous time steps
            cell.dict["Persistence"] = self.Persistence*(np.tanh(self.Sensitivity*F_change)+1.)/2. #map field change into persistence: (-inf, inf)->(0, 1). Sensitivity is tanh slope at origin.
            cell.dict["PT"] = self.PT*(-np.tanh(self.Sensitivity*F_change)+1.)/2. #map field change into PT: (-inf, inf)->(0, 1). Sensitivity is tanh slope at origin.
            '''
            The two previous lines are basically a base parameter value (PT and Persistence) moduled by a sigmoid function. I chose the sigmoid
            function tanh instead of the famous Hill Function because the field grande is restricted to the (-inf, inf) domain, while the Hill
            Function is restricted only to (0, inf) domain (behavior at negative values varies a lot for different exponents). In this case, the
            tanh is a better option since it maps (-inf, inf) to (-1, 1). The image set can be easily adjusted as I did.
            ''' 





################## CONNECTING TO NETOWRK #############################
    
            '''         
            In the following bit of code, we define our state vector, actions that we can take, and update the cell properties accordingly. 
            we then compute the reward for the cell, define our next field change and state and finally update the model. All of this follows a standard, SIMPLE Q-learning algorithm. 
            Specifically, this is known as an epsilon greedy strategy. As in the agent will always choose the action with the highest Q-value. So we have a 'greedy' cell that does 
            whatever gives it the most immediate reward. Because our Force is constant, this often takes the form of the cell moving to the right over and over again. 
            '''
            if not mcs % self.network_timestep:
                field_history = cell.dict["Field_History"]
                
                # define our state vector | field history and field change
                state = np.array(field_history + [F_change])

                '''
                model.predict function take the state as input and returns a list of Q-values, which is then chosen as an action ot perform
                the np.argmax function gives us the index of the action with the highest Q-value which we choose as the next action that the cell performs. 
                '''
                # Choose action |
                action = np.argmax(self.model.predict(state.reshape(1, -1))[0])
    
                # Update cell properties based on action
                self.update_cell_properties(cell, action)
    
                # Compute reward based on cell position in the chemical gradient
                reward = self.compute_reward(cell)
              
              # Add data point for the reward
                self.reward_plot.add_data_point("Reward", mcs, reward)
               
               # Get next state
                next_F_change = F[cell.xCOM, cell.yCOM, 0] - cell.dict["Field_History"][(mcs+2)%10]
                next_state = np.array(cell.dict["Field_History"][1:] + [cell.dict["F_now"], next_F_change])
    
                # Store experience in memory
                self.memory.append((state, action, reward, next_state))
    
                # Update model 
                self.update_model()

                self.plot_win.add_data_point("POS", cell.xCOM, cell.yCOM)
           
            if rd.uniform(0,1) < cell.dict["PT"]: #test if cell will tumble based on PT
            
                #PERSISTENCE
                dtheta = np.random.vonmises(0.,cell.dict["Persistence"]) #pick a random angle using von mises distribution
                cell.dict["P"] += dtheta                                 #updates polarization direction
                    
                #FORCE
                cell.lambdaVecX = -self.Force*np.cos(cell.dict["P"]) #define force x corrdinate (must be negative)
                cell.lambdaVecY = -self.Force*np.sin(cell.dict["P"]) #define force y corrdinate
                cell.lambdaVecZ = 0.0
                
 ################# SAVING MODEL #############################################            
            if mcs == 1000:
                self.model.save('C:/Users/conno/Desktop/test.h5')
            
            if mcs == 30000:
                self.model.save('C:/Users/conno/Desktop/model_weights.h5')
                
            if mcs == 100000:
                self.model.save('C:/Users/conno/Desktop/model_weights_100000mcs.h5')
                
            if mcs == 500000:
                self.model.save('C:/Users/conno/Desktop/model_weights_500000mcs.h5')
            if mcs == 1000000:
                self.model.save('C:/Users/conno/Desktop/model_weights_1000000mcs.h5')    
                
 ##################### DEFINING FUNCTIONS #########################
  
    #function for limiting values for our action update function below
    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
        
    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def update_cell_properties(self, cell, action):
        if action == 0:
            cell.dict["P"] += np.random.vonmises(0., cell.dict["Persistence"])
        elif action == 1:
            cell.dict["P"] -= np.random.vonmises(0., cell.dict["Persistence"])
        elif action == 2:
            cell.dict["Persistence"] *= 1.1
        elif action == 3:
            cell.dict["Persistence"] *= 0.9
        elif action == 4:
            cell.dict['PT'] += 0.1
        elif action == 5:
            cell.dict['PT'] -= 0.1
        
        # Ensure that the values are within a valid range
        cell.dict["P"] = self.wrap_angle(cell.dict["P"])
        cell.dict["Persistence"] = self.clamp(cell.dict["Persistence"], 0, 300)
        cell.dict["PT"] = self.clamp(cell.dict["PT"], 0, 1)    


###### ONLY POSITIVE REWARD ###########


    # def compute_reward(self, cell):
        # F = self.field.Field
        # return F[cell.xCOM, cell.yCOM, 0] / F.max()
  
########## REWARD WITH PENALTY ##########################
        
    def compute_reward(self, cell):
        F = self.field.Field
        x_dim = self.dim.x
        y_dim = self.dim.y
        z_dim = self.dim.z
    
        # Find the maximum field value and its index
        max_field_value = -float("inf")
        max_field_index = (0, 0)
        for x in range(x_dim):
            for y in range(y_dim):
                if F[x, y, 0] > max_field_value:
                    max_field_value = F[x, y, 0]
                    max_field_index = (x, y)
    
        # Calculate the distance from the cell's position to the highest point in the gradient
        distance_to_max = np.sqrt((cell.xCOM - max_field_index[0])**2 + (cell.yCOM - max_field_index[1])**2)
    
        # Scale distance penalty to be in the same range as the reward for being at the highest point
        distance_penalty = distance_to_max / np.sqrt(x_dim**2 + y_dim**2)
    
        #PENTALY 
        penalty_weight = 0.5
    
        #REWARD WITH PENALTY 
        reward = (F[cell.xCOM, cell.yCOM, 0] / max_field_value) - (penalty_weight * distance_penalty)
    
        return reward

################# UPDATING MODEL (GETS CALLED IN STEPPABLES) ########################## 

    def update_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = rd.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Compute target Q-values
        current_q = self.model.predict(states)
        target_q = current_q.copy()
        next_q = self.model.predict(next_states)
        max_next_q = np.max(next_q, axis=1)
       
        for i, (state, action, reward, next_state) in enumerate(batch):
            target_q[i, action] = reward + self.gamma * max_next_q[i]    
        self.model.train_on_batch(states, target_q)        
        
        # Calculate the loss 
        loss = np.mean(np.square(target_q - next_q))
        self.loss_plot.add_data_point("Loss", self.mcs, loss)


############### STEERING PANEL #######################################################

    def load_or_train_model(self):
        
        model_file = 'C:/Users/conno/OneDrive/Desktop/Agent#1/model_weights_500000mcs.h5'
        if self.load_existing_model and os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file)
            print("Loaded existing model.")
        else:
            model = self.create_model()
            print("Created a new model.")
       
        return model
    def handle_steering_panel_data(self):
        # Update the load_existing_model variable based on the Steering Panel input
        self.load_existing_model = self.get_steering_param('load_existing_model') == 'True'
        # Load or create a new model based on the updated load_existing_model value
        self.model = self.load_or_train_model()
 
 
    
    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        return



        