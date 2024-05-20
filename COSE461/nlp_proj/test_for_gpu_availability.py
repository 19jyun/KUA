import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA
from pycuda.driver import Device
import torch

#Last version before merging into multimodal

print("is torch cuda available ", torch.cuda.is_available())
def checkforGPU():
    # Get Id of default device
    print("TensorFlow version:", tf.__version__)

    # Alternatively, to list the physical GPU devices
    print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
    
def print_device_info():
    print("Listing CUDA Devices:\n")
    # Get count of all CUDA Devices
    num_devices = cuda.Device.count()
    
    for i in range(num_devices):
        device = Device(i)
        attributes = device.get_attributes()

        # Extract device attributes
        compute_capability = f"{device.compute_capability_major}.{device.compute_capability_minor}"       
        total_multiprocessors = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]
        cores_per_multiprocessor = _get_cores_per_sm(device.compute_capability_major)
        total_cores = total_multiprocessors * cores_per_multiprocessor
        
        checkforGPU()
        
        print(f"Device {i}: {device.name()}")
        print(f"  Compute Capability: {compute_capability}")
        print(f"  Total Multiprocessors: {total_multiprocessors}")
        print(f"  Cores per Multiprocessor: {cores_per_multiprocessor}")
        print(f"  Total CUDA Cores: {total_cores}")
        print(f"  Total Memory: {device.total_memory() // (1024 ** 2)} MiB\n")

def _get_cores_per_sm(compute_capability_major):
    """Approximate CUDA cores per SM for various architectures."""
    # Cores per SM vary per architecture. Typical numbers used below.
    return {
        1: 8,    # Tesla
        2: 32,   # Fermi
        3: 192,  # Kepler
        5: 128,  # Maxwell
        6: 64,   # Pascal
        7: 64,   # Volta and Turing
        8: 64,   # Ampere
    }.get(compute_capability_major, 64)  # Default to a safe assumption of recent arch

if __name__ == "__main__":
    print_device_info()

