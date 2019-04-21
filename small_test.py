import gym
import time
import gym_toyrpp


env = gym.make('toyrpp-v1000')
env.reset()

steps = 0

while(True):
    env.render()
    ob, reward, game_over, _ = env.step(env.action_space.sample()) # take a random action

    steps += 1
    
    if game_over:
        print ("STEPS ", steps)
        env.reset()    
        steps = 0
        
    time.sleep(5)

env.close()



    

