"""
Simple diagnostic script to check GPU availability.
"""
import sys
import torch
import platform
import subprocess


def get_nvidia_smi_output():
    """Try to run nvidia-smi and return its output."""
    try:
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return output.decode('utf-8')
    except:
        return "nvidia-smi command failed or not found"


def main():
    print("-" * 50)
    print("SYSTEM INFORMATION:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")

    print("\n" + "-" * 50)
    print("CUDA INFORMATION:")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")
    print(f"Number of CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    if torch.cuda.is_available():
        print("\n" + "-" * 50)
        print("GPU INFORMATION:")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")

    print("\n" + "-" * 50)
    print("NVIDIA-SMI OUTPUT:")
    print(get_nvidia_smi_output())

    # Try a simple CUDA operation
    print("\n" + "-" * 50)
    print("TESTING CUDA TENSOR CREATION:")
    try:
        x = torch.rand(5, 3)
        print(f"CPU tensor: {x.device}")

        if torch.cuda.is_available():
            y = torch.rand(5, 3).cuda()
            print(f"GPU tensor: {y.device}")

            # Try a simple operation
            z = y + y
            print(f"Operation result: {z.sum().item()}")
            print("CUDA operations successful")
        else:
            print("Skipping CUDA tensor test as CUDA is not available")
    except Exception as e:
        print(f"Error during CUDA tensor test: {e}")


if __name__ == "__main__":
    main()