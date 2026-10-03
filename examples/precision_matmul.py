import time, torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")

device="cuda"
M=N=K=2048

def bench(dtype):
    a=torch.randn(M,K,device=device,dtype=dtype)
    b=torch.randn(K,N,device=device,dtype=dtype)
    for _ in range(10): a@b
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20): c=a@b
    end.record(); end.synchronize()
    return {"dtype":str(dtype),"avg_ms":start.elapsed_time(end)/20,
            "input_MB":(a.numel()+b.numel())*a.element_size()/1e6,
            "output_shape":tuple(c.shape)}

print("GPU:",torch.cuda.get_device_name())
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    try: print(bench(dtype))
    except RuntimeError as e: print(dtype,"unsupported/error:",e)

print("Interpret results using GPU architecture, Tensor Core eligibility, precision support, and numerical requirements.")
