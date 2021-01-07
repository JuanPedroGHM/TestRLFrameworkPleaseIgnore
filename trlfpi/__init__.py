import pretty_errors

# Configure torch and setup gpu
import torch
torch.set_default_dtype(torch.double)

from gym.envs.registration import register

register(
    id='linear-with-ref-v0',
    entry_point='trlfpi.envs:LinearEnv'
)

register(
    id='clutch-v0',
    entry_point='trlfpi.envs:ClutchEnv'
)
