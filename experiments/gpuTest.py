import torch

print(f'Cuda available: {torch.cuda.is_available()}')

for i in range(torch.cuda.device_count()):
    print(f'Device {i} : {torch.cuda.get_device_name(0)}')
