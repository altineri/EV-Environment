import time
import math
import numpy as np
from scipy.stats import halfnorm, norm
from sklearn.utils import shuffle

import gym
from gym import error, spaces, utils
from gym.utils import seeding


#Action space: Multidimensional. 
#First component is charger type: 0 --> No charging; 1 --> L1 charging; 2 --> L2 charging; 3 --> DC charging; 4 --> xDC charging
#Second component is charging time: {1/4, 1/2, 1, 2,...., 12} --> size 14

#Observation space: 0 --> Battery remaining capacity; 1 --> Energy capacity 
#Energy capacity (remaining charge in kwh) is in [0, 60] continuous interval; Tank size (kwh) is in [0, 60] continuous interval

#Energy consumption rate = 13 - 25.3 kwh per 100km uniform distribution; 100 km = 62.137 miles
#0-12 hr staying at home at the end of the day with mod/median 10hr 
#L1 daily change on battery capacity = 0.0002%; L2 daily change on battery capacity = 0.0004%;
#DC fast change on battery capacity = 0.0006%; Extreme DC fast change on battery capacity = 0.001% of the battery capacity per day. Can compute hourly decrease and connect it with the staying at home hours (0-12hrs)
#Data is from gamma distribution with: Gamma shape parameter --> 1.92; Gamma scale parameter --> 15.20 



class ElectricCarEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.action_space = spaces.Discrete(70)
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([+60.0,+60.0]))
        self.reset()  #--> initial values you are going to give.
 
    
    def reset(self):
        self.tank_size = 60 #kwh 60 kwh corresponds 100% tank_size
        self.energy_capacity = 60 #kwh 60 kwh corresponds 100% charge 
        self.charge = 60
        self.day = 0
        self.done = False
        self.cost = 0
        self.fullchargneed = 0
        np.random.seed(101)
        self.DayLog = np.random.gamma(shape=1.92, scale=15.20, size=365) #how many miles traveled by day. Array of length 365
        self.EConsLog = np.random.uniform(low=13, high=25.3, size=365) #energy consumption rate by day. Array of length 365; kwh/100km
        np.random.seed(101)
        self.mu = 9
        self.sigma = 1.15
        self.X2 = halfnorm.rvs(loc = 0, scale = 0.2, size=20)
        self.X1 = np.random.normal(self.mu, self.sigma, 345)
        self.X = np.concatenate([self.X1, self.X2])
        self.X[self.X < 1] = 0
        self.Y = shuffle(self.X, random_state=101)
        self.state = np.array([self.tank_size, self.energy_capacity], dtype=np.float32)
        
        
        return self.state
    
    
    def after_done_reset(self):
        self.energy_capacity = self.tank_size
        self.state = np.array([self.tank_size, self.energy_capacity], dtype=np.float32)
        
        return self.state
    
    
    def terminated(self):
        
        #print(f'In terminated. D_log:{self.DayLog[self.day]}, E_Clog:{self.EConsLog[self.day]}, Y:{self.Y[self.day]}')

        
        if self.day < 365:
            if (self.energy_capacity < (self.EConsLog[self.day]/62.137)*self.DayLog[self.day]) or self.tank_size == 0:
                return True
            else:
                return False
        elif self.day == 365:
            if self.tank_size == 0:
                return True
            else:
                return False
        else:
            return True
        
           
    
        
    def reward_handler(self,action):
        
        w = 0.5 #weights for battery/cost trade-off

        if action[0] == 1:
            reward = -0.001*action[1]*(1-w)*2 - 1.8*0.0013*action[1]*w*2

        elif action[0] == 2:
            reward = -0.002*action[1]*(1-w)*2 - 7.6*0.0013*action[1]*w*2

        elif action[0] == 3:
            reward = -0.003*action[1]*(1-w)*2 - 60*0.0026*action[1]*w*2


        elif action[0] == 4:
            reward = -0.005*action[1]*(1-w)*2 - 400*0.0052*action[1]*w*2


        elif action[0] == 0:
            reward = 0.001

        
        reward += 50
        
        if self.done and self.day < 365:
            reward -= 50
        
        
        return reward     
    
        
    
    def step(self,action):
        
        action_dict = {0: [0, 0], 1:[0, 1], 2: [0, 2], 3:[0, 3], 4: [0, 4], 5:[0, 5], 6: [0, 6], 7:[0, 7], 8: [0, 8], 9:[0, 9],
                      10: [0, 10], 11:[0, 11], 12: [0, 12], 13:[0, 13], 14: [1, 0], 15:[1, 1], 16: [1, 2], 17:[1, 3], 18: [1, 4], 
                      19:[1, 5], 20: [1, 6], 21:[1, 7], 22: [1, 8], 23:[1, 9], 24: [1, 10], 25:[1, 11], 26: [1, 12], 27:[1, 13], 
                      28: [2, 0], 29:[2, 1], 30: [2, 2], 31:[2, 3], 32: [2, 4], 33:[2, 5], 34: [2, 6], 35:[2, 7], 36: [2, 8], 
                      37:[2, 9], 38: [2, 10], 39:[2, 11], 40: [2, 12], 41:[2, 13], 42: [3, 0], 43: [3, 1], 44: [3, 2], 45: [3, 3], 
                      46: [3, 4], 47: [3, 5], 48:[3, 6], 49: [3, 7], 50:[3, 8], 51: [3, 9], 52:[3, 10], 53:[3, 11], 54:[3, 12], 
                      55:[3, 13], 56:[4, 0], 57: [4, 1], 58:[4, 2], 59: [4, 3], 60:[4, 4], 61: [4, 5], 62:[4, 6], 63:[4, 7], 
                      64:[4, 8], 65:[4, 9], 66:[4, 10], 67:[4, 11], 68:[4, 12], 69:[4, 13]}
        
        #action_dict = [charger type, charging time].
        #[1,0]: [L1, 1 hr], [1, 1]: [L1, 2 hrs] ..., [1, 11]: [L1, 12 hrs], [1, 12]: [L1, 0.5 hr], [1, 13]: [L1, 0.25 hr]
        #[2, 0]: [L2, 1 hr], [2, 1]: [L2, 2 hrs] ..., [2, 11]: [L2, 12 hrs], [2, 12]: [L2, 0.5 hr], [2, 13]: [L2, 0.25 hr]
        #and so on.. 

        original_action = action
        action = action_dict[action]
        
        self.day += 1
        reward = 0
        
        self.energy_capacity = self.energy_capacity - (self.EConsLog[self.day-1]/62.137)*self.DayLog[self.day-1]
        #print(f'In step. D_log:{self.DayLog[self.day]}, E_Clog:{self.EConsLog[self.day]}, Y:{self.Y[self.day]}')
        
        
        if self.energy_capacity <= 0:
            self.energy_capacity = 0
            self.done = True
            
        self.state = np.array([self.tank_size, self.energy_capacity], dtype=np.float32)    
        

        if action[1] == 12 or action[1] == 13:
            or_act = action[1]
            action[1] = abs(action[1]-14)/4
            
            if action[1] > self.Y[self.day-1]: #The case when Y[self.day-1] = 0.
                action[0] = 0 



        else:
            or_act = action[1]
            action[1] = action[1] + 1
            
            if action[1] > self.Y[self.day-1]:
                if self.Y[self.day-1] != 0:
                    action[1] = math.floor(self.Y[self.day-1])
                    or_act = action[1] - 1
                    
                else:
                    action[0] = 0 

                        

        if action[0] == 0: #No charging
            self.done = self.terminated()
            self.cost = 0


        elif action[0] == 1:#L1 charging
            self.fullchargneed = self.tank_size-self.energy_capacity
            if action[1]*1.8 > self.fullchargneed:
                action[1] = self.fullchargneed/1.8
                self.tank_size = self.tank_size - (0.0002/24)*(self.fullchargneed/1.8)*(self.tank_size/100)
                self.energy_capacity = self.tank_size
                self.cost = self.fullchargneed*0.13
            else:
                self.tank_size = self.tank_size - (0.0002/24)*action[1]*(self.tank_size/100)
                self.energy_capacity = self.energy_capacity + 1.8*action[1]
                self.cost = 1.8*0.13*action[1]
                       
            self.done = self.terminated()


        elif action[0] == 2: #L2 charging
            self.fullchargneed = self.tank_size-self.energy_capacity
            if action[1]*7.6 > self.fullchargneed:
                action[1] = self.fullchargneed/7.6
                self.tank_size = self.tank_size - (0.0004/24)*(self.fullchargneed/7.6)*(self.tank_size/100)
                self.energy_capacity = self.tank_size
                self.cost = self.fullchargneed*0.13
            else:
                self.tank_size = self.tank_size - (0.0004/24)*action[1]*(self.tank_size/100)
                self.energy_capacity = self.energy_capacity + 7.6*action[1] 
                self.cost = 7.6*0.13*action[1]
                       
            self.done = self.terminated()
                                    


        elif action[0] == 3: #DC fast charging
            self.fullchargneed = self.tank_size-self.energy_capacity
            if action[1]*60 > self.fullchargneed:
                action[1] = self.fullchargneed/60
                self.tank_size = self.tank_size - (0.0006/24)*(self.fullchargneed/60)*(self.tank_size/100)
                self.energy_capacity = self.tank_size
                self.cost = self.fullchargneed*0.26
            else:
                self.tank_size = self.tank_size - (0.0006/24)*action[1]*(self.tank_size/100)
                self.energy_capacity = self.energy_capacity + 60*action[1] 
                self.cost = 60*0.26*action[1]
                       
            self.done = self.terminated()
               


        elif action[0] == 4: #xDC fast charging
            self.fullchargneed = self.tank_size-self.energy_capacity
            if action[1]*400 > self.fullchargneed:
                action[1] = self.fullchargneed/400
                self.tank_size = self.tank_size - (0.001/24)*(self.fullchargneed/400)*(self.tank_size/100)
                self.energy_capacity = self.tank_size
                self.cost = self.fullchargneed*0.52
            else:
                self.tank_size = self.tank_size - (0.001/24)*action[1]*(self.tank_size/100)
                self.energy_capacity = self.energy_capacity + 400*action[1] 
                self.cost = 400*0.52*action[1]

            self.done = self.terminated()



      
        if self.day == 365:
            self.done = True


                
        self.charge = self.energy_capacity
        done = self.done
        reward = self.reward_handler(action)
        info ={'Day':self.day,'ChargType':action[0]*14+or_act,'OrAct':original_action,'ChargAftCharging':self.charge,'Cost':self.cost}
           
        return self.state, reward, done, info      
    
             
        
    def close(self):
        pass
                    
                
