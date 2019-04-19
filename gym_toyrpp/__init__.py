from gym.envs.registration import register

# registering scenarios here

register(
    id='toyrpp-v0',
    entry_point='gym_toyrpp.envs:ToyRPPEnv'
)
