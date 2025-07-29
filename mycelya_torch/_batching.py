# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
RPC call batching system for improved performance.

This module provides thread-safe batching of RPC calls to reduce network overhead
and improve overall system performance by grouping multiple operations together.
"""

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Union

from ._logging import get_logger

log = get_logger(__name__)


@dataclass
class BatchedRPCCall:
    """
    Represents a single RPC call that has been queued for batching.

    This encapsulates all the information needed to execute an RPC call
    along with metadata for batching and future handling.
    """

    call_type: str  # "spawn" (fire-and-forget) or "remote" (blocking)
    method_name: str  # e.g., "create_storage", "execute_aten_operation"
    args: tuple
    kwargs: dict
    future: Optional[Future] = None  # Only populated for "remote" calls
    queued_at: float = field(default_factory=time.time)  # For cache invalidation timing
    call_id: str = field(
        default_factory=lambda: f"call_{id(object())}"
    )  # Unique ID for debugging


@dataclass
class BatchExecutionResult:
    """
    Result of executing a batch of RPC calls.

    Contains both successful results and any errors that occurred,
    allowing for partial batch execution and proper error handling.
    """

    results: List[Union[Any, Exception]]  # Results in same order as input calls
    success_count: int
    error_count: int
    execution_time: float


class RPCBatchQueue:
    """
    Thread-safe queue for batching RPC calls.

    This queue collects RPC calls from the main thread and allows
    a background thread to process them in batches for improved performance.
    """

    def __init__(self, client_id: str):
        """
        Initialize the batch queue for a specific client.

        Args:
            client_id: Unique identifier for the client (for logging/debugging)
        """
        self.client_id = client_id
        self._queue: Queue[BatchedRPCCall] = Queue()
        self._lock = threading.RLock()  # Re-entrant lock for nested operations

        # Statistics for monitoring
        self._queued_calls = 0
        self._processed_calls = 0
        self._last_batch_time = time.time()

    def enqueue_call(
        self,
        call_type: str,
        method_name: str,
        args: tuple,
        kwargs: dict,
        return_future: bool = False,
    ) -> Optional[Future]:
        """
        Add an RPC call to the batch queue.

        Args:
            call_type: "spawn" for fire-and-forget, "remote" for blocking
            method_name: Name of the RPC method to call
            args: Arguments for the RPC method
            kwargs: Keyword arguments for the RPC method
            return_future: Whether to return a Future for this call

        Returns:
            Future object if return_future=True, None otherwise
        """
        with self._lock:
            future = None
            if return_future or call_type == "remote":
                future = Future()

            call = BatchedRPCCall(
                call_type=call_type,
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                future=future,
            )

            self._queue.put(call)
            self._queued_calls += 1

            log.debug(
                f"📦 Queued {call_type} call: {method_name} for client {self.client_id}"
            )

            return future

    def get_batch(self, timeout: float = 0.01) -> List[BatchedRPCCall]:
        """
        Retrieve all currently queued calls as a batch.

        Args:
            timeout: Maximum time to wait for calls (in seconds)

        Returns:
            List of BatchedRPCCall objects ready for execution
        """
        with self._lock:
            batch = []

            # Get all available calls without blocking
            try:
                while True:
                    call = self._queue.get_nowait()
                    batch.append(call)
                    self._queue.task_done()
            except Empty:
                pass

            if batch:
                self._processed_calls += len(batch)
                self._last_batch_time = time.time()
                log.debug(
                    f"📋 Retrieved batch of {len(batch)} calls for client {self.client_id}"
                )

            return batch

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the batch queue.

        Returns:
            Dictionary containing queue statistics
        """
        with self._lock:
            return {
                "client_id": self.client_id,
                "queue_size": self._queue.qsize(),
                "queued_calls": self._queued_calls,
                "processed_calls": self._processed_calls,
                "last_batch_time": self._last_batch_time,
                "pending_calls": self._queued_calls - self._processed_calls,
            }

    def clear(self) -> None:
        """Clear all pending calls from the queue."""
        with self._lock:
            while not self._queue.empty():
                try:
                    call = self._queue.get_nowait()
                    if call.future:
                        call.future.cancel()
                    self._queue.task_done()
                except Empty:
                    break
            log.info(f"🧹 Cleared batch queue for client {self.client_id}")


class BatchProcessor:
    """
    Processes batched RPC calls and updates futures with results.

    This class handles the execution of batched RPC calls and ensures
    that blocking calls receive their results through the Future mechanism.
    """

    @staticmethod
    def execute_batch(
        server_instance: Any, batch: List[BatchedRPCCall]
    ) -> BatchExecutionResult:
        """
        Execute a batch of RPC calls on the server instance using the batched RPC method.

        Args:
            server_instance: The Modal server instance to execute calls on
            batch: List of batched RPC calls to execute

        Returns:
            BatchExecutionResult containing results and statistics
        """
        if not batch:
            return BatchExecutionResult([], 0, 0, 0.0)

        start_time = time.time()

        log.info(f"🚀 Executing batch of {len(batch)} RPC calls using batched RPC")

        try:
            # Convert BatchedRPCCall objects to the format expected by execute_batch
            batch_calls = []
            for call in batch:
                batch_call = {
                    "method_name": call.method_name,
                    "call_type": call.call_type,
                    "args": call.args,
                    "kwargs": call.kwargs,
                    "call_id": call.call_id,
                }
                batch_calls.append(batch_call)

            log.info(
                f"🔍 DEBUG: About to call server_instance.execute_batch.remote() with {len(batch_calls)} calls"
            )
            log.info(f"🔍 DEBUG: server_instance type: {type(server_instance)}")
            log.info(
                f"🔍 DEBUG: execute_batch method: {type(getattr(server_instance, 'execute_batch', None))}"
            )

            # Execute all calls in a single batched RPC
            results = server_instance.execute_batch.remote(batch_calls)

            log.info(f"🔍 DEBUG: Batched RPC completed, results type: {type(results)}")

            # Update futures with results
            success_count = 0
            error_count = 0

            for _i, (call, result) in enumerate(zip(batch, results)):
                if isinstance(result, Exception):
                    error_count += 1
                    if call.future:
                        call.future.set_exception(result)
                else:
                    success_count += 1
                    if call.future:
                        call.future.set_result(result)

            execution_time = time.time() - start_time

            log.info(
                f"✅ Batch execution complete: {success_count} success, {error_count} errors, {execution_time:.3f}s"
            )

            return BatchExecutionResult(
                results=results,
                success_count=success_count,
                error_count=error_count,
                execution_time=execution_time,
            )

        except Exception as e:
            # If the entire batch fails, update all futures with the exception
            log.error(f"❌ Entire batch failed: {e}")

            for call in batch:
                if call.future:
                    call.future.set_exception(e)

            execution_time = time.time() - start_time

            return BatchExecutionResult(
                results=[e] * len(batch),
                success_count=0,
                error_count=len(batch),
                execution_time=execution_time,
            )
