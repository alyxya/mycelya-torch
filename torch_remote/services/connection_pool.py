# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Connection pool manager for handling client lifecycle and health monitoring.

This service manages connections to remote machines:
- Client lifecycle management (start/stop/restart)
- Health checking and heartbeat tracking
- Connection pooling and reuse
- Automatic reconnection handling

Extracted from RemoteOrchestrator to provide better separation of concerns
and enable advanced connection management features.
"""

import logging
import time
from typing import Dict, Optional

from ..backends.client_interface import ClientInterface
from ..device import RemoteMachine

log = logging.getLogger(__name__)


class ConnectionPoolManager:
    """Manages client connections and their lifecycle.
    
    This class provides centralized management of client connections,
    including health monitoring, reconnection logic, and connection pooling.
    """

    def __init__(self):
        self._last_heartbeat: Dict[str, float] = {}  # Track last successful communication
        self._connection_cache: Dict[str, ClientInterface] = {}  # Cache active connections
        self._heartbeat_threshold = 300.0  # 5 minutes before considering connection stale

    def get_client(self, machine: RemoteMachine) -> ClientInterface:
        """Get an active client for the specified machine.
        
        Args:
            machine: RemoteMachine to get client for
            
        Returns:
            Active ClientInterface instance
            
        Raises:
            RuntimeError: If no client available or client not running
        """
        machine_id = machine.machine_id

        # Check cache first
        if machine_id in self._connection_cache:
            cached_client = self._connection_cache[machine_id]
            if cached_client.is_running():
                self._update_heartbeat(machine_id)
                return cached_client
            else:
                # Remove stale connection from cache
                del self._connection_cache[machine_id]

        # Get fresh client from machine
        client = machine.get_client()
        if client is None:
            raise RuntimeError(f"No client available for machine {machine_id}")

        if not client.is_running():
            raise RuntimeError(f"Client for machine {machine_id} is not running")

        # Cache the active client
        self._connection_cache[machine_id] = client
        self._update_heartbeat(machine_id)

        return client

    def health_check_client(self, machine: RemoteMachine) -> bool:
        """Check if a client connection is healthy.
        
        Args:
            machine: RemoteMachine to check
            
        Returns:
            True if client is healthy, False otherwise
        """
        try:
            client = machine.get_client()
            if client is None:
                return False

            # Check if client reports as running
            if not client.is_running():
                return False

            # Check heartbeat age
            machine_id = machine.machine_id
            last_heartbeat = self._last_heartbeat.get(machine_id)
            if last_heartbeat is not None:
                age = time.time() - last_heartbeat
                if age > self._heartbeat_threshold:
                    log.warning(f"Client {machine_id} heartbeat is stale ({age:.1f}s old)")
                    return False

            return True

        except Exception as e:
            log.warning(f"Health check failed for machine {machine.machine_id}: {e}")
            return False

    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all cached connections.
        
        Returns:
            Dictionary mapping machine_id to health status
        """
        results = {}
        for machine_id in list(self._connection_cache.keys()):
            # We need the RemoteMachine instance to check health
            # For now, just check if cached client is still running
            client = self._connection_cache[machine_id]
            try:
                is_healthy = client.is_running()
                if not is_healthy:
                    # Remove unhealthy connection from cache
                    del self._connection_cache[machine_id]
                results[machine_id] = is_healthy
            except Exception as e:
                log.warning(f"Health check failed for cached client {machine_id}: {e}")
                results[machine_id] = False
                # Remove problematic connection from cache
                del self._connection_cache[machine_id]

        return results

    def reconnect_client(self, machine: RemoteMachine) -> bool:
        """Attempt to reconnect a client.
        
        Args:
            machine: RemoteMachine to reconnect
            
        Returns:
            True if reconnection successful, False otherwise
        """
        machine_id = machine.machine_id

        try:
            # Remove from cache
            self._connection_cache.pop(machine_id, None)

            client = machine.get_client()
            if client:
                # Stop and restart the client
                client.stop()
                client.start()

                # Test connection by checking if it's running
                if client.is_running():
                    log.info(f"Successfully reconnected to machine {machine_id}")
                    self._connection_cache[machine_id] = client
                    self._update_heartbeat(machine_id)
                    return True
                else:
                    log.warning(f"Failed to reconnect to machine {machine_id}")
                    return False
            return False

        except Exception as e:
            log.error(f"Error during reconnection to machine {machine_id}: {e}")
            return False

    def cleanup_stale_connections(self) -> int:
        """Clean up stale connections from the cache.
        
        Returns:
            Number of connections cleaned up
        """
        cleaned_count = 0
        current_time = time.time()

        for machine_id in list(self._connection_cache.keys()):
            # Check heartbeat age
            last_heartbeat = self._last_heartbeat.get(machine_id)
            if last_heartbeat is not None:
                age = current_time - last_heartbeat
                if age > self._heartbeat_threshold:
                    log.info(f"Cleaning up stale connection for {machine_id} (age: {age:.1f}s)")
                    self._connection_cache.pop(machine_id, None)
                    self._last_heartbeat.pop(machine_id, None)
                    cleaned_count += 1

        return cleaned_count

    def close_all_connections(self) -> None:
        """Close all cached connections and clear the cache."""
        for machine_id, client in list(self._connection_cache.items()):
            try:
                if client.is_running():
                    client.stop()
                    log.info(f"Closed connection to {machine_id}")
            except Exception as e:
                log.warning(f"Error closing connection to {machine_id}: {e}")

        self._connection_cache.clear()
        self._last_heartbeat.clear()

    def get_connection_stats(self) -> Dict[str, any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary with connection pool statistics
        """
        current_time = time.time()
        active_connections = len(self._connection_cache)

        heartbeat_ages = {}
        for machine_id, timestamp in self._last_heartbeat.items():
            heartbeat_ages[machine_id] = current_time - timestamp

        return {
            "active_connections": active_connections,
            "heartbeat_threshold": self._heartbeat_threshold,
            "heartbeat_ages": heartbeat_ages,
            "cached_machines": list(self._connection_cache.keys()),
        }

    def _update_heartbeat(self, machine_id: str) -> None:
        """Update the last successful communication timestamp for a machine."""
        self._last_heartbeat[machine_id] = time.time()

    def get_last_heartbeat(self, machine_id: str) -> Optional[float]:
        """Get the timestamp of last successful communication with a machine."""
        return self._last_heartbeat.get(machine_id)
