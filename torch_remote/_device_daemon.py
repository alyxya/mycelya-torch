# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

import atexit
import ctypes
import logging
import os
import signal
import threading
import time
import weakref

import torch

from ._meta_parser import (
    receive_after_sending,
    safe_str,
    validate_send_queue_args,
)


log = logging.getLogger(__name__)

# Use a simple queue-based threading approach instead of multiprocessing
# to avoid the complex cleanup issues with multiprocessing
import queue

# Constant properties of our device
NUM_DEVICES = 2


# Our allocator with tensor ID system
class Allocator:
    def __init__(self):
        self.allocated = {}  # ptr -> (size, mem) for compatibility
        self.tensor_id_counter = 0
        self.tensor_id_to_ptr = {}  # tensor_id -> ptr mapping
        self.ptr_to_tensor_id = {}  # ptr -> tensor_id mapping

    def create_tensor_with_id(self, tensor_id, size):
        """Create tensor with specified ID and return success status"""
        if size == 0:
            # Handle empty tensors - return success for zero-sized allocation
            return True
        
        mem = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(mem)
        
        # Store pointer mapping
        self.allocated[ptr] = (size, mem)
        
        # Store tensor ID mapping
        if isinstance(tensor_id, str):
            # Convert string ID to int if needed (for backward compatibility)
            try:
                tensor_id_int = int(tensor_id.split('_')[1], 16) % (2**63)  # Use hex part
            except:
                self.tensor_id_counter += 1
                tensor_id_int = self.tensor_id_counter
        else:
            tensor_id_int = int(tensor_id)
        
        self.tensor_id_to_ptr[tensor_id_int] = ptr
        self.ptr_to_tensor_id[ptr] = tensor_id_int
        
        return True

    def malloc(self, size):
        """Legacy malloc for compatibility"""
        mem = ctypes.create_string_buffer(size)
        ptr = ctypes.addressof(mem)
        self.allocated[ptr] = (size, mem)
        return ptr

    def free_tensor_with_id(self, tensor_id):
        """Free tensor by ID"""
        tensor_id_int = int(tensor_id)
        if tensor_id_int == 0:  # Empty tensor
            return True
            
        if tensor_id_int not in self.tensor_id_to_ptr:
            return False
            
        ptr = self.tensor_id_to_ptr[tensor_id_int]
        
        # Clean up mappings
        del self.tensor_id_to_ptr[tensor_id_int]
        del self.ptr_to_tensor_id[ptr]
        
        # Clean up allocated memory
        if ptr in self.allocated:
            del self.allocated[ptr]
            
        return True

    def free(self, ptr):
        """Legacy free for compatibility"""
        if ptr not in self.allocated:
            return False
        else:
            # Clean up tensor ID mappings if they exist
            if ptr in self.ptr_to_tensor_id:
                tensor_id = self.ptr_to_tensor_id[ptr]
                del self.tensor_id_to_ptr[tensor_id]
                del self.ptr_to_tensor_id[ptr]
            
            del self.allocated[ptr]
            return True




class DeviceAllocator(Allocator):
    def tensor_from_meta(self, meta):
        def create_tensor_from_data_ptr(ptr, size):
            storage = torch._C._construct_storage_from_data_pointer(
                ptr, torch.device("cpu"), size
            )
            return torch.Tensor(storage)

        found_base = None
        
        # Check if meta.data_ptr is a tensor ID (likely if it's a small integer)
        tensor_id = meta.data_ptr
        
        # Handle empty tensors
        if meta.nelem_in_bytes == 0:
            found_base = torch.tensor((), dtype=torch.uint8)
        
        # Try to find by tensor ID first
        elif tensor_id in self.tensor_id_to_ptr:
            ptr = self.tensor_id_to_ptr[tensor_id]
            if ptr in self.allocated:
                found_base = create_tensor_from_data_ptr(
                    ptr, self.allocated[ptr][0]
                )
        
        # Fallback: treat as legacy pointer (for compatibility)
        elif meta.data_ptr in self.allocated:
            found_base = create_tensor_from_data_ptr(
                meta.data_ptr, self.allocated[meta.data_ptr][0]
            )

        # Might be a rewrap of another storage at a different offset (legacy)
        if found_base is None:
            for tag, (size, _) in self.allocated.items():
                # t is always a 1D uint8 storage!
                if meta.data_ptr > tag and meta.data_ptr < tag + size:
                    # Blame @ngimel for this
                    slice_size = size - (meta.data_ptr - tag)
                    found_base = create_tensor_from_data_ptr(meta.data_ptr, slice_size)

        # This tensor ID is not allocated here, error!
        if found_base is None:
            log.info("Currently allocated blocks:\n %s", safe_str(self.allocated))
            log.info("Tensor ID mappings:\n %s", safe_str(self.tensor_id_to_ptr))
            log.info("Trying to access %s", meta)
            raise RuntimeError(f"TENSOR ID {tensor_id} NOT FOUND!")

        # Raw 1d uint8 data
        raw = found_base
        # Reinterpret cast in the right dtype
        as_dtype = raw.view(dtype=meta.dtype)
        # View to the right shape/stride/offset
        view = as_dtype.as_strided(meta.size, meta.stride, meta.storage_offset)
        
        # Preserve requires_grad property if available
        if hasattr(meta, 'requires_grad') and meta.requires_grad:
            view.requires_grad_(True)
        
        return view


def register(registry):
    def func(fn):
        registry[fn.__name__] = fn
        return fn

    return func


class Driver:
    _instances = weakref.WeakSet()
    _signal_handlers_registered = False

    def __init__(self, num_devices):
        super().__init__()
        self.num_devices = num_devices
        self.is_initialized = False

        # State of our driver
        self.curr_device_idx = 0
        self.curr_streams = {}

        # Allocated memory belongs to which device
        self.memory_belong = {}
        self.event_belong = {}
        
        # Tensor ID to metadata mapping for efficient tensor operations
        self.tensor_id_to_meta = {}  # tensor_id -> RemoteTensorMeta
        self.tensor_id_to_tensor = {}  # tensor_id -> torch.Tensor (weak references)

        self.rlock = threading.RLock()
        
        # Register this instance for cleanup
        Driver._instances.add(self)
        
        # Register signal handlers once
        if not Driver._signal_handlers_registered:
            Driver._register_signal_handlers()
            Driver._signal_handlers_registered = True

    def _lazy_init(self):
        if self.is_initialized:
            return
        self.devices = []

        # Start with initial number of devices
        for i in range(self.num_devices):
            self._add_device(i)

        self.is_initialized = True
        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _add_device(self, device_id):
        """Add a new device executor."""
        req_queue = queue.Queue()
        ans_queue = queue.Queue()
        executor = _Executor(device_id)
        
        # Use threading instead of multiprocessing
        runner = threading.Thread(
            target=executor.run_forever,
            args=(req_queue, ans_queue),
            daemon=True  # Daemon threads will exit when main process exits
        )
        runner.start()
        self.devices.append((req_queue, ans_queue, runner, executor))

    @classmethod
    def _register_signal_handlers(cls):
        """Register signal handlers for proper cleanup."""
        def signal_handler(signum, frame):
            cls._cleanup_all_instances()
            # Re-raise the signal to continue normal termination
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)
        
        # Register for common termination signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            try:
                signal.signal(sig, signal_handler)
            except (OSError, ValueError):
                # Signal handling might not be available in some contexts
                pass

    @classmethod
    def _cleanup_all_instances(cls):
        """Clean up all Driver instances."""
        for instance in list(cls._instances):
            try:
                instance._cleanup()
            except Exception:
                pass

    def _cleanup(self):
        """Clean up daemon processes on exit."""
        if not self.is_initialized:
            return
        
        with self.rlock:
            # Signal all threads to shutdown
            for req_queue, ans_queue, runner, executor in self.devices:
                if runner.is_alive():
                    try:
                        # Send shutdown signal
                        req_queue.put(("shutdown",), block=False)
                    except:
                        pass
            
            # Wait for threads to finish
            for req_queue, ans_queue, runner, executor in self.devices:
                if runner.is_alive():
                    runner.join(timeout=0.1)
            
            self.devices = []
            self.is_initialized = False
            
            # Also cleanup library registrations
            try:
                from ._aten_impl import cleanup_library_registrations
                cleanup_library_registrations()
            except Exception:
                pass

    def exec(self, cmd, *args):
        with self.rlock:
            log.info("Main process launched: %s(*%s)", cmd, safe_str(args))

            if cmd in Driver.registry:
                res = Driver.registry[cmd](self, *args)
            else:
                res = self.run_on_executor(self.curr_device_idx, cmd, *args)

            log.info("Main process result for %s received: %s", cmd, safe_str(res))
            if res == "ERROR":
                raise RuntimeError(f"Error in daemon while executing {cmd}, see logs")
            else:
                return res

    def run_on_executor(self, device_idx, cmd, *args):
        self._lazy_init()
        # Ensure we have enough devices for the requested index
        while device_idx >= len(self.devices):
            self._add_device(len(self.devices))
        req_queue, ans_queue, _, _ = self.devices[device_idx]
        stream = self.getStream(device_idx)
        validate_send_queue_args(cmd, args)
        req_queue.put((stream, cmd) + args)
        return ans_queue.get()

    registry = {}

    @register(registry)
    def hasPrimaryContext(self, device_idx):
        # We can dynamically create devices, so any non-negative index is valid
        return device_idx >= 0

    @register(registry)
    def deviceCount(self, *args):
        assert len(args) == 0
        self._lazy_init()
        return len(self.devices)

    @register(registry)
    def getDevice(self):
        return self.curr_device_idx

    @register(registry)
    def setDevice(self, device_idx):
        assert device_idx >= 0
        self._lazy_init()
        # Ensure we have enough devices for the requested index
        while device_idx >= len(self.devices):
            self._add_device(len(self.devices))
        self.curr_device_idx = device_idx

    @register(registry)
    def uncheckedSetDevice(self, *args):
        assert len(args) == 1
        self.curr_device_idx = int(args[0])

    @register(registry)
    def exchangeDevice(self, *args):
        assert len(args) == 1
        res = self.curr_device_idx
        self.curr_device_idx = int(args[0])
        return res

    @register(registry)
    def create_tensor_with_id(self, tensor_id, nbytes, device_index):
        """Create a tensor with the given ID and return success status"""
        # Route to the specified device
        success = self.run_on_executor(device_index, "create_tensor_with_id", tensor_id, nbytes)
        
        # Track which device owns this tensor ID (for cleanup)
        if success and nbytes > 0:  # Don't track empty tensors
            self.memory_belong[int(tensor_id)] = device_index
            
        return success
    
    @register(registry)
    def register_tensor_mapping(self, tensor_id, tensor, meta):
        """Register a tensor ID to tensor and metadata mapping"""
        tensor_id_int = int(tensor_id)
        self.tensor_id_to_meta[tensor_id_int] = meta
        # Store weak reference to avoid circular references
        import weakref
        self.tensor_id_to_tensor[tensor_id_int] = weakref.ref(tensor)
    
    @register(registry)
    def get_tensor_by_id(self, tensor_id):
        """Get tensor by its ID"""
        tensor_id_int = int(tensor_id)
        if tensor_id_int in self.tensor_id_to_tensor:
            tensor_ref = self.tensor_id_to_tensor[tensor_id_int]
            return tensor_ref()  # Dereference weak reference
        return None
    
    @register(registry)
    def get_meta_by_id(self, tensor_id):
        """Get tensor metadata by its ID"""
        tensor_id_int = int(tensor_id)
        return self.tensor_id_to_meta.get(tensor_id_int)

    @register(registry)
    def free_tensor_with_id(self, tensor_id):
        """Free tensor by tensor ID"""
        tensor_id_int = int(tensor_id)
        if tensor_id_int == 0:  # Empty tensor
            return True
            
        device_idx = self.memory_belong.pop(tensor_id_int, None)
        if device_idx is None:
            return False
        
        # Clean up tensor mappings
        self.tensor_id_to_meta.pop(tensor_id_int, None)
        self.tensor_id_to_tensor.pop(tensor_id_int, None)
        
        return self.run_on_executor(device_idx, "free_tensor_with_id", tensor_id_int)

    @register(registry)
    def copy_data_by_id(self, dest_id, src_id, count):
        """Copy data between tensors identified by their IDs"""
        # For now, this is a placeholder - actual copy implementation would need
        # to look up the memory locations for both tensor IDs and perform the copy
        dest_device = self.memory_belong.get(int(dest_id))
        src_device = self.memory_belong.get(int(src_id))
        
        if dest_device is None or src_device is None:
            raise RuntimeError(f"Copy failed: tensor IDs {dest_id} or {src_id} not found")
        
        # For now, assume same-device copy (actual implementation would handle cross-device)
        if dest_device == src_device:
            return self.run_on_executor(dest_device, "copy_data_by_id", dest_id, src_id, count)
        else:
            raise RuntimeError("Cross-device copy not yet implemented")

    @register(registry)
    def malloc(self, size):
        """Legacy malloc for compatibility"""
        ptr = self.run_on_executor(self.curr_device_idx, "malloc", size)
        self.memory_belong[ptr] = self.curr_device_idx
        return ptr

    @register(registry)
    def free(self, ptr):
        """Legacy free for compatibility"""
        device_idx = self.memory_belong.pop(ptr, None)
        if device_idx is None:
            return False
        return self.run_on_executor(device_idx, "free", ptr)


    @register(registry)
    def getNewStream(self, device_idx, priority):
        return self.run_on_executor(device_idx, "getNewStream", priority)

    @register(registry)
    def queryStream(self, stream):
        return self.run_on_executor(
            stream.device_index, "queryStream", stream.stream_id
        )

    @register(registry)
    def getStream(self, device_idx):
        return self.curr_streams.get(device_idx, 0)

    @register(registry)
    def exchangeStream(self, stream):
        stream_id = self.curr_streams.get(stream.device_index, 0)
        self.curr_streams[stream.device_index] = stream.stream_id
        return stream_id

    @register(registry)
    def synchronizeStream(self, stream):
        self.run_on_executor(stream.device_index, "synchronizeStream", stream.stream_id)

    @register(registry)
    def record(self, event, stream, device_index, flags):
        event_ptr = ctypes.cast(event, ctypes.POINTER(ctypes.c_int64))
        # Create event if needed
        if event_ptr.contents.value == 0:
            event_ptr.contents.value = self.run_on_executor(
                stream.device_index, "eventCreateWithFlags", flags
            )
            self.event_belong[event_ptr.contents.value] = stream.device_index

        # Record event
        self.run_on_executor(
            stream.device_index,
            "eventRecord",
            event_ptr.contents.value,
            stream.stream_id,
        )

    @register(registry)
    def destroyEvent(self, event, device_index):
        self.run_on_executor(device_index, "eventDestroy", event)
        self.event_belong.pop(event)

    @register(registry)
    def synchronizeEvent(self, event):
        self.run_on_executor(self.event_belong[event], "eventSynchronize", event)

    @register(registry)
    def queryEvent(self, event):
        return self.run_on_executor(self.event_belong[event], "eventQuery", event)

    @register(registry)
    def elapsedTime(self, e1, e2, device_index):
        return self.run_on_executor(device_index, "eventElapsedTime", e1, e2)

    @register(registry)
    def block(self, event, stream):
        self.run_on_executor(stream.device_index, "block", event, stream.stream_id)


class _Executor:
    def __init__(self, id):
        self.id = id
        self.allocator = DeviceAllocator()
        self.stream = 0
        self.event_incr_id = 0
        self.events = {}

    def run_forever(self, req_queue, ans_queue):
        # Serve all requests
        while True:
            # Ignore stream since cpu backend doesn't support asynchronous execution
            req = req_queue.get()
            if req[0] == "shutdown":
                break
            _, cmd, *args = req
            log.info("Worker executing: %s", cmd)
            if cmd in _Executor.registry:
                res = _Executor.registry[cmd](self, *args)
            else:
                log.warning("Bad command in worker")
                res = "ERROR"

            log.info("Worker answering to: %s", cmd)
            ans_queue.put(res)

    registry = {}

    @register(registry)
    def create_tensor_with_id(self, tensor_id, size):
        """Create tensor with ID in executor"""
        return self.allocator.create_tensor_with_id(tensor_id, size)

    @register(registry)
    def free_tensor_with_id(self, tensor_id):
        """Free tensor by ID in executor"""
        return self.allocator.free_tensor_with_id(tensor_id)

    @register(registry)
    def copy_data_by_id(self, dest_id, src_id, count):
        """Copy data between tensors by ID in executor"""
        dest_id_int = int(dest_id)
        src_id_int = int(src_id)
        
        # Look up the memory pointers for both tensors
        if (dest_id_int not in self.allocator.tensor_id_to_ptr or 
            src_id_int not in self.allocator.tensor_id_to_ptr):
            return False
            
        dest_ptr = self.allocator.tensor_id_to_ptr[dest_id_int]
        src_ptr = self.allocator.tensor_id_to_ptr[src_id_int]
        
        # Get the memory buffers and perform the copy
        dest_size, dest_mem = self.allocator.allocated[dest_ptr]
        src_size, src_mem = self.allocator.allocated[src_ptr]
        
        # Ensure we don't copy more than available
        copy_size = min(count, dest_size, src_size)
        ctypes.memmove(dest_mem, src_mem, copy_size)
        return True

    @register(registry)
    def malloc(self, size):
        """Legacy malloc for compatibility"""
        return self.allocator.malloc(size)

    @register(registry)
    def free(self, ptr):
        """Legacy free for compatibility"""
        return self.allocator.free(ptr)

    def _run_op(self, op_name, args, kwargs):
        op, _ = torch._C._jit_get_operation(op_name)
        args, kwargs = receive_after_sending(self.allocator, args, kwargs)
        return op(*args, **kwargs)

    @register(registry)
    def run_op(self, op_name, args, kwargs):
        self._run_op(op_name, args, kwargs)

    @register(registry)
    def get_op_output_shape(self, op_name, args, kwargs):
        return self._run_op(op_name, args, kwargs).size()

    @register(registry)
    def send_data(self, *args):
        assert len(args) == 1
        return self.allocator.tensor_from_meta(args[0])

    @register(registry)
    def recv_data(self, host_tensor, dev_mem):
        dev_tensor = self.allocator.tensor_from_meta(dev_mem)
        dev_tensor.copy_(host_tensor)

    @register(registry)
    def getNewStream(self, priority):
        self.stream += 1
        return self.stream

    @register(registry)
    def queryStream(self, stream):
        return True

    @register(registry)
    def synchronizeStream(self, stream):
        # no-op
        pass

    @register(registry)
    def eventCreateWithFlags(self, flags):
        self.event_incr_id += 1
        self.events[self.event_incr_id] = [flags, None]
        return self.event_incr_id

    @register(registry)
    def eventRecord(self, event, stream):
        # Only flags == 1 enables timing
        if self.events[event][0] == 1:
            self.events[event][1] = time.time() * 1000
        return 0

    @register(registry)
    def eventDestroy(self, event):
        self.events.pop(event)

    @register(registry)
    def eventSynchronize(self, event):
        assert self.events.get(event) is not None
        return 0

    @register(registry)
    def eventQuery(self, event):
        assert self.events.get(event) is not None
        return True

    @register(registry)
    def eventElapsedTime(self, e1, e2):
        time_1 = self.events[e1][1]
        time_2 = self.events[e2][1]
        assert time_1 is not None and time_2 is not None
        return time_2 - time_1

    @register(registry)
    def block(self, event, stream):
        # no-op
        pass

    @register(registry)
    def empty_tensor(self, size, dtype):
        # Calculate default strides for contiguous tensor
        if len(size) == 0:
            # Scalar tensor (0-dimensional) has empty strides
            stride = []
        else:
            stride = [1]
            for i in range(len(size) - 2, -1, -1):
                stride.insert(0, stride[0] * size[i + 1])
        
        return self.empty_strided_tensor(size, stride, dtype)

    @register(registry)
    def empty_strided_tensor(self, size, stride, dtype):
        # Allocate memory for the tensor
        element_size = torch.zeros(1, dtype=dtype).element_size()
        
        # Handle scalar tensors (0-dimensional) separately
        if len(size) == 0:
            # Scalar tensor always needs one element
            total_size = element_size
        else:
            # For non-scalar tensors, calculate size from strides
            total_size = max(s * st for s, st in zip(size, stride)) * element_size if size else 0
        
        if total_size > 0:
            ptr = self.allocator.malloc(total_size)
        else:
            # For empty tensors, use a dummy pointer
            ptr = 0
        
        # Create a meta object to represent this tensor
        from ._meta_parser import RemoteTensorMeta
        
        # Create a minimal tensor meta
        class TensorMeta:
            def __init__(self, data_ptr, size, stride, storage_offset, dtype, nelem_in_bytes):
                self.data_ptr = data_ptr
                self.size = size
                self.stride = stride
                self.storage_offset = storage_offset
                self.dtype = dtype
                self.nelem_in_bytes = nelem_in_bytes
        
        nelem = 1
        for s in size:
            nelem *= s
        nelem_in_bytes = nelem * element_size
        
        meta = TensorMeta(ptr, size, stride, 0, dtype, nelem_in_bytes)
        return meta


driver = Driver(NUM_DEVICES)

# Register global cleanup for all contexts
atexit.register(Driver._cleanup_all_instances)

# Note: pytest cleanup is handled in conftest.py to prevent hanging