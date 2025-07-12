import pprint

import torch
from torch.utils._pytree import tree_map, tree_map_only


class RemoteTensorMeta:
    def __init__(self, tensor, checked=True):
        if checked and not tensor.device.type == "remote":
            raise RuntimeError(
                "Creating RemoteTensorMeta is only for Tensors on remote device"
            )
        self.data_ptr = tensor.untyped_storage().data_ptr()
        self.size = tensor.size()
        self.stride = tensor.stride()
        self.storage_offset = tensor.storage_offset()
        self.dtype = tensor.dtype
        self.nelem_in_bytes = tensor.nelement() * tensor.element_size()

    def __repr__(self):
        return (
            f"RemoteTensorMeta({self.data_ptr=}, {self.size=}, {self.stride=}, "
            f"{self.storage_offset=}, {self.dtype=}, {self.nelem_in_bytes=})"
        )


class RemoteTensorData(torch.Tensor):
    @staticmethod
    def from_meta(allocator, tensor_meta):
        return RemoteTensorData(allocator.tensor_from_meta(tensor_meta))
    
    @property
    def device(self):
        """Override device property to report 'remote' instead of the underlying CPU device."""
        import torch
        # Get the device index from the BackendDevice via the device ID
        device_id = getattr(self, '_device_id', None)
        if device_id is not None:
            from .device import get_device_registry
            registry = get_device_registry()
            backend_device = registry.get_device_by_id(device_id)
            if backend_device is not None:
                device_index = backend_device.remote_index
                return torch.device("remote", device_index)
        
        raise RuntimeError(f"RemoteTensorData missing device ID")
    
    def cpu(self, memory_format=torch.preserve_format):
        """Override cpu() method to return actual CPU tensor, not RemoteTensorData."""
        # Get the underlying tensor data and create a true CPU tensor
        cpu_tensor = super().cpu()
        # Ensure it's a regular torch.Tensor, not RemoteTensorData
        return torch.tensor(cpu_tensor.detach().numpy())
    
    def __str__(self):
        """Safe string representation for remote tensors."""
        try:
            # Handle scalar tensors (0-dimensional)
            if self.ndim == 0:
                shape_str = "()"
            else:
                shape_str = str(tuple(self.shape))
            return f"RemoteTensor(shape={shape_str}, dtype={self.dtype}, device={self.device})"
        except Exception:
            return f"RemoteTensor(device=remote)"
    
    def __repr__(self):
        """Safe repr representation for remote tensors."""
        try:
            # Handle scalar tensors (0-dimensional)
            if self.ndim == 0:
                shape_str = "()"
            else:
                shape_str = str(tuple(self.shape))
            return f"RemoteTensor(shape={shape_str}, dtype={self.dtype}, device={self.device})"
        except Exception:
            return f"RemoteTensor(device=remote)"
    
    @classmethod
    def _validate_mixed_device_operation(cls, func, args, kwargs):
        """Prevent operations between tensors on different devices."""
        import torch
        
        devices = set()
        
        # Check all tensor arguments for device types
        def check_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                devices.add(tensor.device)
        
        # Check args
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    check_tensor(item)
            else:
                check_tensor(arg)
        
        # Check kwargs if present
        if kwargs:
            for value in kwargs.values():
                if isinstance(value, (list, tuple)):
                    for item in value:
                        check_tensor(item)
                else:
                    check_tensor(value)
        
        # Raise error if multiple different devices detected
        if len(devices) > 1:
            op_name = func._qualified_name if hasattr(func, '_qualified_name') else str(func)
            device_list = [str(d) for d in sorted(devices, key=str)]
            raise RuntimeError(
                f"Cannot perform {op_name} between tensors on different devices: {device_list}. "
                "Move all tensors to the same device first using .to()."
            )
    
    @classmethod  
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Dispatch remote tensor operations to remote execution when appropriate.
        This properly integrates with PyTorch's dispatch system.
        """
        if kwargs is None:
            kwargs = {}
        
        # Get operation name
        op_name = func._qualified_name if hasattr(func, '_qualified_name') else str(func)
        
        # Skip dispatch for certain operations to avoid infinite recursion
        skip_dispatch_ops = {
            'aten.alias.default',
            'aten._make_subclass.default', 
            'aten.detach.default',
            'aten.requires_grad_.default',
            'aten.is_contiguous.memory_format',
            'aten.size.default',
            'aten.stride.default',
            'aten.storage_offset.default',
            'aten.numel.default',
            'aten.dim.default',
            'aten.copy_.default'
        }
        
        # Skip validation for tensor creation operations (single-device by nature)
        skip_validation_ops = {
            # Basic factory functions
            'aten.empty', 'aten.empty.memory_format', 'aten.empty.default',
            'aten.empty_like', 'aten.empty_strided',
            'aten.zeros', 'aten.zeros.default', 'aten.zeros_like',
            'aten.ones', 'aten.ones.default', 'aten.ones_like',
            'aten.full', 'aten.full_like',
            
            # Random number generation
            'aten.randn', 'aten.randn.default', 'aten.rand', 'aten.rand_like',
            'aten.randn_like', 'aten.randint', 'aten.randint_like', 'aten.randperm',
            'aten.normal', 'aten.normal_.default', 'aten.uniform', 'aten.uniform_',
            'aten.random_',
            
            # Sequential and range functions
            'aten.arange', 'aten.arange.start', 'aten.arange.start_step',
            'aten.linspace', 'aten.logspace',
            
            # Special matrices
            'aten.eye', 'aten.eye.m',
            
            # Tensor construction from data
            'aten.tensor', 'aten.lift_fresh', 'aten.lift_fresh.default',
            'aten.scalar_tensor',
            
            # Initialization functions
            'aten.new_empty', 'aten.new_zeros', 'aten.new_ones', 'aten.new_full'
        }
        
        if op_name in skip_dispatch_ops:
            # Use torch.Tensor's default behavior for these operations
            return torch.Tensor.__torch_dispatch__(func, types, args, kwargs)
        
        # Optionally log for debugging - comment out for production
        # print(f"🔍 RemoteTensorData.__torch_dispatch__ called for {op_name}")
        
        # Skip validation for tensor creation operations (single-device by nature)
        if op_name not in skip_validation_ops:
            # Validate mixed device operations before proceeding
            cls._validate_mixed_device_operation(func, args, kwargs)
        
        # Import here to avoid circular imports
        from ._aten_impl import _should_use_remote_execution, _get_remote_executor
        
        # Create a mock op object for _should_use_remote_execution
        class MockOp:
            class MockOverloadPacket:
                def __init__(self, name):
                    self._qualified_op_name = name
            
            def __init__(self, name):
                self.overloadpacket = MockOp.MockOverloadPacket(op_name)
        
        mock_op = MockOp(op_name)
        
        # Check if we should use remote execution
        if _should_use_remote_execution(mock_op, args, kwargs):
            executor = _get_remote_executor()
            if executor is not None:
                # print(f"🚀 Using remote execution for {op_name}")
                try:
                    return executor.execute_remote_operation(op_name, args, kwargs)
                except Exception as e:
                    # print(f"⚠️  Remote execution failed for {op_name}: {e}, falling back to CPU")
                    pass  # Silently fall back to CPU
        
        # Fallback to CPU execution
        # print(f"🔄 Using CPU fallback for {op_name}")
        
        # Convert remote tensors to CPU for fallback execution
        def to_cpu_if_remote(arg):
            if isinstance(arg, RemoteTensorData):
                # Use torch.Tensor's behavior directly to get underlying data
                return torch.Tensor.__torch_dispatch__(torch.ops.aten.detach.default, (torch.Tensor,), (arg,), {})
            return arg
        
        cpu_args = tuple(to_cpu_if_remote(arg) for arg in args)
        cpu_kwargs = {k: to_cpu_if_remote(v) for k, v in kwargs.items()}
        
        # Execute on CPU using torch.Tensor's dispatch
        cpu_result = torch.Tensor.__torch_dispatch__(func, types, cpu_args, cpu_kwargs)
        
        # Convert result back to remote tensor if it's a tensor
        if isinstance(cpu_result, torch.Tensor):
            return cpu_result.to("remote")
        return cpu_result


VALID_QUEUE_TYPES_IN = {torch.Tensor, int, float, torch.dtype}

VALID_QUEUE_TYPES_OUT = {RemoteTensorMeta, int, float, str, torch.dtype}


def safe_str(args):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return str(RemoteTensorMeta(obj, checked=False))
        else:
            return obj

    new_args = tree_map(convert, args)
    return pprint.pformat(new_args)


def validate_send_queue_args(cmd, args):
    def check(obj):
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            if (
                cmd == "recv_data"
                and type(obj) in [torch.Tensor, RemoteTensorData]
                and obj.device.type == "cpu"
            ):
                # Only HtoD copy command can send cpu Tensors over
                return
            raise RuntimeError(
                f"Trying to send invalid object through queue: {type(obj)}"
            )

    tree_map(check, args)


def prepare_for_sending(args, kwargs):
    def convert(obj):
        if type(obj) not in VALID_QUEUE_TYPES_IN:
            raise RuntimeError(
                f"Cannot send object of type {type(obj)} over remote device pipe."
            )

        if isinstance(obj, torch.Tensor):
            return RemoteTensorMeta(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


def receive_after_sending(allocator, args, kwargs):
    def convert(obj):
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            raise RuntimeError(
                f"Received invalid object of type {type(obj)} over remote device pipe."
            )

        if isinstance(obj, RemoteTensorMeta):
            return allocator.tensor_from_meta(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


def to_device_no_copy(device, args, kwargs):
    def safe_to(t):
        if device == "meta":
            return t.to(device=device)
        else:
            return torch.empty_like(t, device=device)

    return tree_map_only(torch.Tensor, safe_to, (args, kwargs))