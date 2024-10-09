import torch
import tensorflow as tf
import os
import numpy as np
import sys

def check_gpu_pytorch():
    print("Pytorch: \n ------------------------")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get the current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"GPU device name: {device_name}")
        
        # Memory information
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_reserved = torch.cuda.memory_reserved(current_device)
        print(f"Memory allocated: {memory_allocated/1024**2:.2f} MB")
        print(f"Memory reserved: {memory_reserved/1024**2:.2f} MB")
        
        # Simple tensor operation to verify GPU usage
        print("\nTesting GPU with a simple tensor operation...")
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        print("GPU tensor operation completed successfully!")
        
    else:
        print("No CUDA device available. Your PyTorch installation is CPU-only.")


def check_gpu_tensorflow():
    print("Tensorflow: \n ------------------------")
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # List all physical devices
    print("\nPhysical devices:")
    print(tf.config.list_physical_devices())
    
    # List specifically GPU devices
    print("\nGPU devices:")
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    
    if gpus:
        # Get GPU device details
        for gpu in gpus:
            print(f"\nGPU Details:")
            print(f"Name: {gpu.name}")
            print(f"Device type: {gpu.device_type}")
            
        # Try to perform a simple operation on GPU
        print("\nTesting GPU with a simple tensor operation...")
        with tf.device('/GPU:0'):
            # Create random matrices
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Perform matrix multiplication
            c = tf.matmul(a, b)
            
            # Force execution (TensorFlow is lazy by default)
            result = c.numpy()
            
        print("GPU tensor operation completed successfully!")
        
        # Show memory info
        print("\nGPU Memory details:")
        print(tf.config.experimental.get_memory_info('GPU:0'))
        
    else:
        print("\nNo GPU devices found. Your TensorFlow installation is CPU-only.")
        
    # Print compute capability if using CUDA
    if tf.test.is_built_with_cuda():
        print("\nTensorFlow is built with CUDA")
    else:
        print("\nTensorFlow is NOT built with CUDA")
    
    # Show if TensorFlow can see the GPU
    print(f"\nTensorFlow able to see GPU: {tf.test.is_gpu_available()}")
    
    # Memory growth settings
    print("\nMemory growth settings:")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"Memory growth setting failed for {gpu.name}: {e}")


if __name__ == "__main__":
    check_gpu_pytorch()
    check_gpu_tensorflow()
