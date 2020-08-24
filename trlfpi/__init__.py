from gym.envs.registration import register

register(
    id='linear-with-ref-v0',
    entry_point='trlfpi.envs:LinearSystemEnv'
)
