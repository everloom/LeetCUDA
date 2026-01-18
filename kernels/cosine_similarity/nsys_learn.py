# test.py
import torch
from torch.cuda import nvtx

with nvtx.range("init"):
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    a = torch.zeros(233, device='cuda')

with torch.cuda.stream(stream1):
    with nvtx.range("first add"):
        a += 1

with torch.cuda.stream(stream2):
    with nvtx.range("second add"):
        a += 1

with nvtx.range("print"):
    torch.cuda.synchronize()
    average = a.mean().item()
    print(f'Average of the elements in tensor a: {average}')