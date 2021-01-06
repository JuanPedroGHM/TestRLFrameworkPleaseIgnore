import pretty_errors
from gym.envs.registration import register


register(
    id='linear-with-ref-v0',
    entry_point='trlfpi.envs:LinearEnv'
)

register(
    id='clutch-v0',
    entry_point='trlfpi.envs:ClutchEnv'
)
