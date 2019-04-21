from gym.envs.registration import register

# registering scenarios here

register(
    id='toyrpp-v0',
    entry_point='gym_toyrpp.envs:ToyRPPEnv'
)


register(
    id='toyrpp-v1000',
    entry_point='gym_toyrpp.envs:ToyRPPEnv',
    kwargs={'env_size':1000}
)


