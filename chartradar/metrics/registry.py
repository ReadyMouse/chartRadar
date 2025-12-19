"""
Algorithm registry and discovery system for the ChartRadar framework.

This module provides functionality to register, discover, and lookup
algorithms by name with support for versioning.
"""

from typing import Any, Dict, List, Optional, Type
import importlib
import inspect
from pathlib import Path

from chartradar.metrics.base import Algorithm
from chartradar.core.exceptions import AlgorithmNotFoundError


class AlgorithmRegistry:
    """
    Registry for pattern detection algorithms.
    
    Provides algorithm registration, discovery, and lookup functionality
    with support for versioning.
    """
    
    def __init__(self):
        """Initialize the algorithm registry."""
        self._algorithms: Dict[str, Dict[str, Type[Algorithm]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        algorithm_class: Type[Algorithm],
        name: Optional[str] = None,
        version: Optional[str] = None,
        overwrite: bool = False
    ) -> None:
        """
        Register an algorithm class.
        
        Args:
            algorithm_class: Algorithm class to register
            name: Algorithm name (uses class name if None)
            version: Algorithm version (uses class version if None)
            overwrite: Whether to overwrite existing registration
            
        Raises:
            ValueError: If algorithm is already registered and overwrite is False
        """
        # Get algorithm name and version
        if name is None:
            name = algorithm_class.__name__
        if version is None:
            # Try to get version from class
            version = getattr(algorithm_class, 'version', '1.0.0')
        
        # Initialize name entry if needed
        if name not in self._algorithms:
            self._algorithms[name] = {}
            self._metadata[name] = {}
        
        # Check if version already exists
        if version in self._algorithms[name] and not overwrite:
            raise ValueError(
                f"Algorithm '{name}' version '{version}' is already registered. "
                "Use overwrite=True to replace it."
            )
        
        # Register algorithm
        self._algorithms[name][version] = algorithm_class
        
        # Store metadata
        try:
            # Create a temporary instance to get metadata
            instance = algorithm_class(name=name, version=version)
            self._metadata[name][version] = instance.get_metadata()
        except Exception:
            # If metadata retrieval fails, store basic info
            self._metadata[name][version] = {
                "name": name,
                "version": version,
                "class": algorithm_class.__name__
            }
    
    def get_algorithm(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Type[Algorithm]:
        """
        Get an algorithm class by name and optional version.
        
        Args:
            name: Algorithm name
            version: Algorithm version (uses latest if None)
            
        Returns:
            Algorithm class
            
        Raises:
            AlgorithmNotFoundError: If algorithm is not found
        """
        if name not in self._algorithms:
            raise AlgorithmNotFoundError(
                f"Algorithm '{name}' not found in registry",
                details={"name": name, "available": list(self._algorithms.keys())}
            )
        
        versions = self._algorithms[name]
        
        if version:
            if version not in versions:
                raise AlgorithmNotFoundError(
                    f"Algorithm '{name}' version '{version}' not found",
                    details={
                        "name": name,
                        "version": version,
                        "available_versions": list(versions.keys())
                    }
                )
            return versions[version]
        else:
            # Return latest version (highest version string)
            if not versions:
                raise AlgorithmNotFoundError(
                    f"No versions found for algorithm '{name}'",
                    details={"name": name}
                )
            # Simple version comparison (assumes semantic versioning)
            latest_version = max(versions.keys(), key=lambda v: self._version_key(v))
            return versions[latest_version]
    
    def _version_key(self, version: str) -> tuple:
        """
        Convert version string to tuple for comparison.
        
        Args:
            version: Version string (e.g., "1.2.3")
            
        Returns:
            Tuple of integers for comparison
        """
        try:
            parts = version.split('.')
            return tuple(int(p) for p in parts)
        except (ValueError, AttributeError):
            # If version parsing fails, return (0, 0, 0)
            return (0, 0, 0)
    
    def create_algorithm(
        self,
        name: str,
        version: Optional[str] = None,
        **kwargs: Any
    ) -> Algorithm:
        """
        Create an algorithm instance.
        
        Args:
            name: Algorithm name
            version: Algorithm version (uses latest if None)
            **kwargs: Parameters to pass to algorithm constructor
            
        Returns:
            Algorithm instance
        """
        algorithm_class = self.get_algorithm(name, version)
        return algorithm_class(name=name, version=version or "1.0.0", **kwargs)
    
    def list_algorithms(self) -> List[str]:
        """
        List all registered algorithm names.
        
        Returns:
            List of algorithm names
        """
        return list(self._algorithms.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """
        List all versions for an algorithm.
        
        Args:
            name: Algorithm name
            
        Returns:
            List of version strings
        """
        if name not in self._algorithms:
            raise AlgorithmNotFoundError(
                f"Algorithm '{name}' not found",
                details={"name": name}
            )
        return list(self._algorithms[name].keys())
    
    def get_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for an algorithm.
        
        Args:
            name: Algorithm name
            version: Algorithm version (uses latest if None)
            
        Returns:
            Metadata dictionary
        """
        if name not in self._metadata:
            raise AlgorithmNotFoundError(
                f"Algorithm '{name}' not found",
                details={"name": name}
            )
        
        versions = self._metadata[name]
        
        if version:
            if version not in versions:
                raise AlgorithmNotFoundError(
                    f"Algorithm '{name}' version '{version}' not found",
                    details={"name": name, "version": version}
                )
            return versions[version]
        else:
            # Return latest version metadata
            latest_version = max(versions.keys(), key=lambda v: self._version_key(v))
            return versions[latest_version]
    
    def discover_algorithms(
        self,
        package_path: str,
        algorithm_base_class: Type[Algorithm] = Algorithm
    ) -> int:
        """
        Discover and register algorithms from a package.
        
        Args:
            package_path: Python package path (e.g., 'chartradar.metrics.algorithms')
            algorithm_base_class: Base class to search for
            
        Returns:
            Number of algorithms discovered and registered
        """
        count = 0
        
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent if hasattr(package, '__file__') else None
            
            if package_dir:
                # Walk through package directory
                for file_path in package_dir.rglob('*.py'):
                    if file_path.name == '__init__.py':
                        continue
                    
                    # Import module
                    module_name = file_path.stem
                    relative_path = file_path.relative_to(package_dir.parent)
                    module_path = str(relative_path.with_suffix('')).replace('/', '.')
                    
                    try:
                        module = importlib.import_module(module_path)
                        
                        # Find algorithm classes
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, algorithm_base_class) and
                                obj is not algorithm_base_class):
                                try:
                                    self.register(obj)
                                    count += 1
                                except Exception as e:
                                    # Skip if registration fails
                                    continue
                    except Exception:
                        # Skip modules that can't be imported
                        continue
        
        except Exception:
            # If discovery fails, return count so far
            pass
        
        return count
    
    def unregister(self, name: str, version: Optional[str] = None) -> None:
        """
        Unregister an algorithm.
        
        Args:
            name: Algorithm name
            version: Algorithm version (removes all versions if None)
        """
        if name not in self._algorithms:
            return
        
        if version:
            if version in self._algorithms[name]:
                del self._algorithms[name][version]
                if version in self._metadata[name]:
                    del self._metadata[name][version]
                
                # Clean up if no versions left
                if not self._algorithms[name]:
                    del self._algorithms[name]
                    del self._metadata[name]
        else:
            # Remove all versions
            del self._algorithms[name]
            del self._metadata[name]


# Global registry instance
_default_registry = AlgorithmRegistry()


def register_algorithm(
    name: Optional[str] = None,
    version: Optional[str] = None,
    registry: Optional[AlgorithmRegistry] = None
):
    """
    Decorator to register an algorithm class.
    
    Usage:
        @register_algorithm(name="my_algorithm", version="1.0.0")
        class MyAlgorithm(Algorithm):
            ...
    
    Args:
        name: Algorithm name (uses class name if None)
        version: Algorithm version
        registry: Registry to use (uses default if None)
    """
    def decorator(cls: Type[Algorithm]):
        reg = registry or _default_registry
        reg.register(cls, name=name, version=version)
        return cls
    return decorator


def get_algorithm(name: str, version: Optional[str] = None) -> Type[Algorithm]:
    """
    Get an algorithm class from the default registry.
    
    Args:
        name: Algorithm name
        version: Algorithm version (uses latest if None)
        
    Returns:
        Algorithm class
    """
    return _default_registry.get_algorithm(name, version)


def create_algorithm(name: str, version: Optional[str] = None, **kwargs: Any) -> Algorithm:
    """
    Create an algorithm instance from the default registry.
    
    Args:
        name: Algorithm name
        version: Algorithm version (uses latest if None)
        **kwargs: Parameters to pass to algorithm constructor
        
    Returns:
        Algorithm instance
    """
    return _default_registry.create_algorithm(name, version, **kwargs)

