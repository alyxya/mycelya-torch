import pprint

import torch
from torch.utils._pytree import tree_map, tree_map_only


class ModalTensorMeta:
    def __init__(self, tensor, checked=True):
        if checked and not tensor.device.type == "modal":
            raise RuntimeError(
                "Creating ModalTensorMeta is only for Tensors on modal device"
            )
        self.data_ptr = tensor.untyped_storage().data_ptr()
        self.size = tensor.size()
        self.stride = tensor.stride()
        self.storage_offset = tensor.storage_offset()
        self.dtype = tensor.dtype
        self.nelem_in_bytes = tensor.nelement() * tensor.element_size()

    def __repr__(self):
        return (
            f"ModalTensorMeta({self.data_ptr=}, {self.size=}, {self.stride=}, "
            f"{self.storage_offset=}, {self.dtype=}, {self.nelem_in_bytes=})"
        )


class ModalTensorData(torch.Tensor):
    @staticmethod
    def from_meta(allocator, tensor_meta):
        return ModalTensorData(allocator.tensor_from_meta(tensor_meta))
    
    @property
    def device(self):
        """Override device property to report 'modal' instead of the underlying CPU device."""
        import torch
        # Get current modal device index, default to 0 if not available
        try:
            device_index = torch.modal.current_device()
        except (AttributeError, RuntimeError):
            device_index = 0
        return torch.device("modal", device_index)
    
    def cpu(self, memory_format=torch.preserve_format):
        """Override cpu() method to return actual CPU tensor, not ModalTensorData."""
        # Get the underlying tensor data and create a true CPU tensor
        cpu_tensor = super().cpu()
        # Ensure it's a regular torch.Tensor, not ModalTensorData
        return torch.tensor(cpu_tensor.detach().numpy())
    
    @classmethod  
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Dispatch modal tensor operations to remote execution when appropriate.
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
        
        if op_name in skip_dispatch_ops:
            # Use torch.Tensor's default behavior for these operations
            return torch.Tensor.__torch_dispatch__(func, types, args, kwargs)
        
        # Optionally log for debugging - comment out for production
        # print(f"🔍 ModalTensorData.__torch_dispatch__ called for {op_name}")
        
        # Import here to avoid circular imports
        from ._aten_impl import _REMOTE_EXECUTION_ENABLED, _should_use_remote_execution, _get_remote_executor
        
        # Create a mock op object for _should_use_remote_execution
        class MockOp:
            class MockOverloadPacket:
                def __init__(self, name):
                    self._qualified_op_name = name
            
            def __init__(self, name):
                self.overloadpacket = MockOp.MockOverloadPacket(op_name)
        
        mock_op = MockOp(op_name)
        
        # Check if we should use remote execution
        if _REMOTE_EXECUTION_ENABLED and _should_use_remote_execution(mock_op, args, kwargs):
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
        
        # Convert modal tensors to CPU for fallback execution
        def to_cpu_if_modal(arg):
            if isinstance(arg, ModalTensorData):
                # Use torch.Tensor's behavior directly to get underlying data
                return torch.Tensor.__torch_dispatch__(torch.ops.aten.detach.default, (torch.Tensor,), (arg,), {})
            return arg
        
        cpu_args = tuple(to_cpu_if_modal(arg) for arg in args)
        cpu_kwargs = {k: to_cpu_if_modal(v) for k, v in kwargs.items()}
        
        # Execute on CPU using torch.Tensor's dispatch
        cpu_result = torch.Tensor.__torch_dispatch__(func, types, cpu_args, cpu_kwargs)
        
        # Convert result back to modal tensor if it's a tensor
        if isinstance(cpu_result, torch.Tensor):
            return cpu_result.to("modal")
        return cpu_result


VALID_QUEUE_TYPES_IN = {torch.Tensor, int, float, torch.dtype}

VALID_QUEUE_TYPES_OUT = {ModalTensorMeta, int, float, str, torch.dtype}


def safe_str(args):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return str(ModalTensorMeta(obj, checked=False))
        else:
            return obj

    new_args = tree_map(convert, args)
    return pprint.pformat(new_args)


def validate_send_queue_args(cmd, args):
    def check(obj):
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            if (
                cmd == "recv_data"
                and type(obj) in [torch.Tensor, ModalTensorData]
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
                f"Cannot send object of type {type(obj)} over modal device pipe."
            )

        if isinstance(obj, torch.Tensor):
            return ModalTensorMeta(obj)
        else:
            return obj

    return tree_map(convert, (args, kwargs))


def receive_after_sending(allocator, args, kwargs):
    def convert(obj):
        if type(obj) not in VALID_QUEUE_TYPES_OUT:
            raise RuntimeError(
                f"Received invalid object of type {type(obj)} over modal device pipe."
            )

        if isinstance(obj, ModalTensorMeta):
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