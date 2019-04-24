import gym
import numpy as np
import copy
import cv2
from gym import error, spaces, utils
from gym.utils import seeding

VIEWPORT = (16, 16)
UPSCALING = 4

class ToyRPPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def _get_new_environment(self):
        """ Initializes the environment with random obstacles and target, starting position
        
        Obstacles = 1
        Open Air = 0

        """
        
        self.current_env = np.random.rand(self.env_size, self.env_size)

        # Build the environment
        self.current_env[self.current_env > self.obs_threshold] = 1
        self.current_env[self.current_env <= self.obs_threshold] = 0
        self.current_pos = np.random.randint(0, self.env_size, 2)

        # Check if the starting position is an obstacle
        # and take another one if it is
        while(self.current_env[self.current_pos[0],
                               self.current_pos[1]] == 1.0):
            self.current_pos = np.random.randint(0, self.env_size, 2)
            
        self.target_pos = np.random.randint(0, self.env_size, 2)

        while(self.current_env[self.target_pos[0],
                               self.target_pos[1]] == 1.0):
            self.target_pos = np.random.randint(0, self.env_size, 2)
        
        
    def _get_obs(self):
        """
        build an rgb  image with the information and store it in the buffer
        """

        my_viewport_x = max(0, self.current_pos[0] - VIEWPORT[0]/2)
        my_viewport_x = min(my_viewport_x, self.env_size - VIEWPORT[0])

        my_viewport_y = max(0, self.current_pos[1] - VIEWPORT[1]/2)
        my_viewport_y = min(my_viewport_y, self.env_size - VIEWPORT[1])

        my_viewport_x = int(my_viewport_x)
        my_viewport_y = int(my_viewport_y)


#        print ("MY_VIEWPORT_X ", my_viewport_x, "Y ", my_viewport_y, self.current_pos)
        
        new_env_view = copy.deepcopy(self.current_env)

        new_env_view[self.current_pos[0], self.current_pos[1]] = 2
        new_env_view[self.target_pos[0], self.target_pos[1]] = 3

        self._buffer = np.zeros((VIEWPORT[0], VIEWPORT[1], 3), dtype=np.uint8)
        self._buffer[new_env_view[my_viewport_x:my_viewport_x + VIEWPORT[0],
                                  my_viewport_y:my_viewport_y + VIEWPORT[1]] == 1] = (125, 0, 0)

        self._buffer[new_env_view[my_viewport_x:my_viewport_x + VIEWPORT[0],
                                  my_viewport_y:my_viewport_y + VIEWPORT[1]] == 2] = (0, 255, 0)

        self._buffer[new_env_view[my_viewport_x:my_viewport_x + VIEWPORT[0],
                                  my_viewport_y:my_viewport_y + VIEWPORT[1]] == 3] = (0, 0, 200)

        # FIXME: delete
        #self._buffer[0:4,0, 2 ] = (int(255 *self.current_pos[0]/self.env_size),
        #                           int(255 *self.current_pos[1]/self.env_size),
        #                           int(255 *self.target_pos[0]/self.env_size),
        #                           int(255 *self.target_pos[1]/self.env_size))

        return cv2.resize(self._buffer, (UPSCALING * VIEWPORT[0], UPSCALING * VIEWPORT[1])) # FIXME: scale
                                       
    def __init__(self, env_size=64, obs_threshold=0.9, my_seed=None):
        """ Initialize the environment 
        Parameters:
        - env_size : (width, height) of the environment

        """

        screen_width, screen_height = VIEWPORT

        # to ensure compatibility
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(screen_height * UPSCALING, screen_width * UPSCALING, 3))

        # buffer with them image
        self._buffer = np.empty((screen_height, screen_width, 3),
                                dtype=np.uint8)
        
        self.viewer = None

        # Must do before
        self._seed(my_seed)


        self.action_space = spaces.Discrete(4)
        
        self.current_env = None

        # store the size for initiliazing on reset
        self.env_size = env_size
        self.obs_threshold = obs_threshold
        self.max_steps = env_size * 4
        # the game will finish if the agent exceeds the maximum number of steps
        self.current_step = 0
        self.game_over = False

        self.env_size = env_size
        

    def _act(self, action):

        my_act = ACTION_LOOKUP[action]

        prev_pos = copy.deepcopy(self.current_pos)
        if my_act == "UP":
            self.current_pos[1] += 1 
            if self.current_pos[1] == self.env_size:
                self.current_pos[1] -= 1

        elif my_act == "DOWN":
            self.current_pos[1] -= 1
            if self.current_pos[1] < 0:
                self.current_pos[1] = 0
                
        elif my_act == "LEFT":
            self.current_pos[0] -= 1
            if self.current_pos[0] < 0:
                self.current_pos[0] = 0
            
        elif my_act == "RIGHT":
            self.current_pos[0] += 1

            if self.current_pos[0] == self.env_size:
                self.current_pos[0] -= 1
                
        self.current_step += 1

        
        if self.current_step > self.max_steps:
            self.game_over = True
            print ("Game Over:Maxsteps")
            

        if self.current_pos[0] == self.target_pos[0] and self.current_pos[1] == self.target_pos[1]:
            print ("Win")
 
        if self.current_env[self.current_pos[0],
                            self.current_pos[1]] == 1:
            self.game_over = True
        
        reward = self._get_reward(prev_pos)

        return reward, self.game_over

    def _get_reward(self, prev_pos):
        """ Obtains the reward regarding the current action peformed by the agent 

        NOTE: Modify this function to get your custom reward for this environment

        parameter:
        prev_pos: previous position of the agent

        """

        if self.current_env[self.current_pos[0], self.current_pos[1]] == 1:
            return -1.0

        if self.current_pos[0] == self.target_pos[0] and self.current_pos[1] == self.target_pos[1]:
            # New target pos
            self.target_pos = np.random.randint(0, self.env_size, 2)

            while(self.current_env[self.target_pos[0],
                                   self.target_pos[1]] == 1.0):
                self.target_pos = np.random.randint(0, self.env_size, 2)

            print ("MAX Reward ")
                
            return 1000.0

        elif (np.linalg.norm(self.current_pos - self.target_pos) <
              np.linalg.norm(prev_pos - self.target_pos)):
            return .01

        else:
            # the agent is not getting closer to the target
            return -.04

    def step(self, action):
        """ Executes one action step and returns:
        - ob: current observation as image
        - reward: reward as given by the environment
        - game_over: flag that marks the episode has finished
        - dictionary with:
          - agent_position: position after performing the action
          - target_postion: the goal of the agent

        """
        
        reward, game_over = self._act(action)
        
        ob = self._get_obs()

        if reward > 1:
            print ("GAME OVER ? ", game_over, "REWARD", reward)
            
        
        return ob, reward, game_over, {"agent_pos": self.current_pos,
                                       "target_pos": self.target_pos}
        
    def reset(self):

        self.current_step = 0
        self.game_over = False
        # init the environment

        self._get_new_environment()
        return self._get_obs()

    def render(self, mode='human'):

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        self.viewer.imshow(self._get_obs())        

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _seed(self, my_seed=None):
        """ Initializing the seed """
        self.np_random, seed1 = seeding.np_random(my_seed)
    
    @property
    def _n_actions(self):
        return len(ACTION_LOOKUP.Keys())

        
ACTION_LOOKUP = {
    0 : "UP",
    1 : "DOWN",
    2 : "LEFT",
    3 : "RIGHT",

}
