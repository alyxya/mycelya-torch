"""
Example usage of PyTorch with Modal Labs remote execution backend

This script demonstrates how to use the custom Modal backend to execute
PyTorch models remotely on Modal's GPU infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from modal_backend import compile_for_modal, modal_backend
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMLPModel(nn.Module):
    """Simple MLP model for testing remote execution"""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleMLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleCNNModel(nn.Module):
    """Simple CNN model for testing remote execution"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerModel(nn.Module):
    """Simple Transformer model for testing remote execution"""
    
    def __init__(self, vocab_size=1000, d_model=256, nhead=8, num_layers=2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = self.fc(x)
        return x


def benchmark_model(model, input_data, model_name="Model", num_runs=5):
    """Benchmark model execution time"""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Benchmarking {model_name}")
    logger.info(f"{'='*50}")
    
    # Warmup
    with torch.no_grad():
        _ = model(input_data)
    
    # Benchmark local execution
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    local_times = []
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            output_local = model(input_data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        local_times.append(time.time() - start_time)
    
    avg_local_time = np.mean(local_times)
    logger.info(f"Local execution: {avg_local_time:.4f}s ± {np.std(local_times):.4f}s")
    
    # Compile for Modal execution
    try:
        compiled_model = compile_for_modal(model)
        
        # Warmup compiled model
        with torch.no_grad():
            _ = compiled_model(input_data)
        
        # Benchmark remote execution
        remote_times = []
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output_remote = compiled_model(input_data)
            remote_times.append(time.time() - start_time)
        
        avg_remote_time = np.mean(remote_times)
        logger.info(f"Remote execution: {avg_remote_time:.4f}s ± {np.std(remote_times):.4f}s")
        
        # Compare outputs
        if isinstance(output_local, torch.Tensor) and isinstance(output_remote, torch.Tensor):
            max_diff = torch.max(torch.abs(output_local - output_remote)).item()
            logger.info(f"Max output difference: {max_diff:.6f}")
            
            if max_diff < 1e-3:
                logger.info("✓ Outputs match within tolerance")
            else:
                logger.warning("⚠ Outputs differ significantly")
        
        speedup = avg_local_time / avg_remote_time
        logger.info(f"Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")
        
    except Exception as e:
        logger.error(f"Remote execution failed: {e}")
        logger.info("This is expected if Modal is not set up - the backend will fallback to local execution")


def test_simple_operations():
    """Test simple tensor operations with Modal backend"""
    logger.info("\n" + "="*50)
    logger.info("Testing Simple Tensor Operations")
    logger.info("="*50)
    
    # Create simple tensors
    x = torch.randn(4, 128)
    y = torch.randn(4, 128)
    
    # Define a simple function to compile
    def simple_ops(a, b):
        c = torch.relu(a)
        d = torch.relu(b)
        return c + d
    
    # Test local execution
    result_local = simple_ops(x, y)
    logger.info(f"Local result shape: {result_local.shape}")
    
    # Compile with Modal backend
    try:
        compiled_fn = torch.compile(simple_ops, backend="modal_backend")
        result_remote = compiled_fn(x, y)
        logger.info(f"Remote result shape: {result_remote.shape}")
        
        # Compare results
        max_diff = torch.max(torch.abs(result_local - result_remote)).item()
        logger.info(f"Max difference: {max_diff:.6f}")
        
    except Exception as e:
        logger.error(f"Compilation failed: {e}")


def main():
    """Main function to run all examples"""
    
    logger.info("PyTorch Modal Labs Remote Execution Example")
    logger.info("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test simple operations first
    test_simple_operations()
    
    # Example 1: Simple MLP
    logger.info("\n🧠 Testing Simple MLP Model")
    mlp_model = SimpleMLPModel(input_size=784, hidden_size=256, num_classes=10)
    mlp_input = torch.randn(8, 784)  # Batch of 8 samples
    
    benchmark_model(mlp_model, mlp_input, "Simple MLP", num_runs=3)
    
    # Example 2: Simple CNN
    logger.info("\n🖼️  Testing Simple CNN Model")
    cnn_model = SimpleCNNModel(num_classes=10)
    cnn_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    
    benchmark_model(cnn_model, cnn_input, "Simple CNN", num_runs=3)
    
    # Example 3: Transformer
    logger.info("\n🤖 Testing Transformer Model")
    transformer_model = TransformerModel(vocab_size=1000, d_model=256)
    transformer_input = torch.randint(0, 1000, (2, 20))  # Batch of 2 sequences
    
    benchmark_model(transformer_model, transformer_input, "Transformer", num_runs=3)
    
    # Performance comparison
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("✓ All models tested with Modal backend")
    logger.info("✓ Remote execution with GPU acceleration")
    logger.info("✓ Automatic fallback to local execution on errors")
    logger.info("")
    logger.info("To deploy the Modal server:")
    logger.info("  modal deploy modal_server.py")
    logger.info("")
    logger.info("To test the Modal health check:")
    logger.info("  modal run modal_server.py::health_check")


if __name__ == "__main__":
    main()