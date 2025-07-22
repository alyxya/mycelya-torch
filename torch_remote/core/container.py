# Copyright (C) 2025 alyxya
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Dependency injection container for torch_remote services.

This module provides a simple dependency injection container to manage
service instances and their dependencies, reducing circular imports and
improving testability.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, TypeVar

log = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class ServiceRegistration:
    """Registration information for a service."""
    service_class: Type
    factory: Optional[Callable] = None
    singleton: bool = True
    instance: Optional[Any] = None


class ServiceContainer:
    """Simple dependency injection container for torch_remote services."""

    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._instances: Dict[Type, Any] = {}

    def register(self,
                 service_type: Type[T],
                 implementation: Type[T] = None,
                 factory: Callable[[], T] = None,
                 singleton: bool = True) -> None:
        """Register a service with the container.
        
        Args:
            service_type: The interface/base type to register for
            implementation: The concrete implementation class (optional if factory provided)
            factory: Factory function to create instances (optional)
            singleton: Whether to reuse the same instance (default: True)
        """
        if implementation is None and factory is None:
            implementation = service_type

        self._services[service_type] = ServiceRegistration(
            service_class=implementation,
            factory=factory,
            singleton=singleton
        )

        # Clear any cached instance
        if service_type in self._instances:
            del self._instances[service_type]

        log.debug(f"Registered service {service_type.__name__} -> {implementation.__name__ if implementation else 'factory'}")

    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register a pre-created instance.
        
        Args:
            service_type: The service type
            instance: The instance to register
        """
        self._services[service_type] = ServiceRegistration(
            service_class=type(instance),
            singleton=True,
            instance=instance
        )
        self._instances[service_type] = instance
        log.debug(f"Registered instance {service_type.__name__}")

    def get(self, service_type: Type[T]) -> T:
        """Get an instance of the requested service.
        
        Args:
            service_type: The service type to get
            
        Returns:
            Instance of the requested service
            
        Raises:
            ValueError: If service is not registered
        """
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} is not registered")

        registration = self._services[service_type]

        # Return pre-registered instance if available
        if registration.instance is not None:
            return registration.instance

        # Check if we have a cached singleton instance
        if registration.singleton and service_type in self._instances:
            return self._instances[service_type]

        # Create new instance
        if registration.factory:
            instance = registration.factory()
        else:
            # Use default constructor
            instance = registration.service_class()

        # Cache singleton instances
        if registration.singleton:
            self._instances[service_type] = instance

        log.debug(f"Created instance of {service_type.__name__}")
        return instance

    def resolve_dependencies(self, service_class: Type[T]) -> T:
        """Create an instance with automatic dependency resolution.
        
        This is a basic implementation that works for simple constructor injection.
        For more complex scenarios, consider using a full DI framework.
        
        Args:
            service_class: The class to instantiate with dependencies
            
        Returns:
            Instance with resolved dependencies
        """
        # For now, just use the simple get() method
        # In the future, this could inspect constructor parameters
        # and automatically resolve them
        return self.get(service_class)

    def clear(self) -> None:
        """Clear all registered services and instances."""
        self._services.clear()
        self._instances.clear()
        log.debug("Cleared all services from container")

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered.
        
        Args:
            service_type: The service type to check
            
        Returns:
            True if registered, False otherwise
        """
        return service_type in self._services

    def get_registered_services(self) -> Dict[str, str]:
        """Get information about registered services.
        
        Returns:
            Dictionary mapping service names to implementation names
        """
        return {
            service_type.__name__: (
                registration.service_class.__name__
                if registration.service_class
                else "factory"
            )
            for service_type, registration in self._services.items()
        }


# Global container instance
_container = ServiceContainer()


def get_container() -> ServiceContainer:
    """Get the global service container."""
    return _container


def register_service(service_type: Type[T],
                    implementation: Type[T] = None,
                    factory: Callable[[], T] = None,
                    singleton: bool = True) -> None:
    """Register a service with the global container."""
    _container.register(service_type, implementation, factory, singleton)


def register_instance(service_type: Type[T], instance: T) -> None:
    """Register an instance with the global container."""
    _container.register_instance(service_type, instance)


def get_service(service_type: Type[T]) -> T:
    """Get a service from the global container."""
    return _container.get(service_type)


def clear_container() -> None:
    """Clear the global container."""
    _container.clear()


# Service registration function for bootstrap
def register_default_services() -> None:
    """Register default torch_remote services with the container."""
    from ..services.storage_resolver import StorageMachineResolver
    from ..services.tensor_transfer import TensorTransferService

    # Register core services as singletons
    register_service(TensorTransferService, singleton=True)
    register_service(StorageMachineResolver, singleton=True)

    log.info("Registered default torch_remote services")
