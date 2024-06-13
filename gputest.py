import torch

# Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    print("CUDA is available!")

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get the name of the current GPU
    current_gpu = torch.cuda.get_device_name(0)
    print(f"Current GPU: {current_gpu}")

    # Get remaining memory
    current_mem = torch.cuda.mem_get_info()
    print(f"Available memory: {current_mem}")

    # Create a tensor and move it to the GPU
    current_gpu = torch.cuda.get_device_name(0)
    print(f"Current GPU: {current_gpu}")

    tensor = torch.randn(3, 4)
    tensor = tensor.to("cuda")
    print(f"Tensor on GPU: {tensor}")

    # Perform some computation on the GPU
    result = tensor.matmul(tensor.t())
    print(f"Result of computation on GPU: {result}")

else:
    print("CUDA is not available. Using CPU instead.")
