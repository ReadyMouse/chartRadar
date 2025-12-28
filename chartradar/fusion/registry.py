"""
Fusion strategy registry for the ChartRadar framework.

This module provides functionality to register and lookup fusion strategies.
"""

from typing import Any, Dict, List, Optional, Type
import importlib
import inspect
from pathlib import Path

from chartradar.fusion.base import FusionStrategy
from chartradar.src.exceptions import FusionStrategyNotFoundError


class FusionStrategyRegistry:
    """
    Registry for fusion strategies.
    
    Provides strategy registration and lookup functionality.
    """
    
    def __init__(self):
        """Initialize the fusion strategy registry."""
        self._strategies: Dict[str, Dict[str, Type[FusionStrategy]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        strategy_class: Type[FusionStrategy],
        name: Optional[str] = None,
        version: Optional[str] = None,
        overwrite: bool = False
    ) -> None:
        """
        Register a fusion strategy class.
        
        Args:
            strategy_class: Fusion strategy class to register
            name: Strategy name (uses class name if None)
            version: Strategy version (uses class version if None)
            overwrite: Whether to overwrite existing registration
            
        Raises:
            ValueError: If strategy is already registered and overwrite is False
        """
        # Get strategy name and version
        if name is None:
            name = strategy_class.__name__
        if version is None:
            version = getattr(strategy_class, 'version', '1.0.0')
        
        # Initialize name entry if needed
        if name not in self._strategies:
            self._strategies[name] = {}
            self._metadata[name] = {}
        
        # Check if version already exists
        if version in self._strategies[name] and not overwrite:
            raise ValueError(
                f"Fusion strategy '{name}' version '{version}' is already registered. "
                "Use overwrite=True to replace it."
            )
        
        # Register strategy
        self._strategies[name][version] = strategy_class
        
        # Store metadata
        try:
            # Create a temporary instance to get metadata
            instance = strategy_class(name=name, version=version)
            self._metadata[name][version] = instance.get_metadata()
        except Exception:
            # If metadata retrieval fails, store basic info
            self._metadata[name][version] = {
                "name": name,
                "version": version,
                "class": strategy_class.__name__
            }
    
    def get_strategy(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Type[FusionStrategy]:
        """
        Get a fusion strategy class by name and optional version.
        
        Args:
            name: Strategy name
            version: Strategy version (uses latest if None)
            
        Returns:
            Fusion strategy class
            
        Raises:
            FusionStrategyNotFoundError: If strategy is not found
        """
        if name not in self._strategies:
            raise FusionStrategyNotFoundError(
                f"Fusion strategy '{name}' not found in registry",
                details={"name": name, "available": list(self._strategies.keys())}
            )
        
        versions = self._strategies[name]
        
        if version:
            if version not in versions:
                raise FusionStrategyNotFoundError(
                    f"Fusion strategy '{name}' version '{version}' not found",
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
                raise FusionStrategyNotFoundError(
                    f"No versions found for fusion strategy '{name}'",
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
    
    def create_strategy(
        self,
        name: str,
        version: Optional[str] = None,
        **kwargs: Any
    ) -> FusionStrategy:
        """
        Create a fusion strategy instance.
        
        Args:
            name: Strategy name
            version: Strategy version (uses latest if None)
            **kwargs: Parameters to pass to strategy constructor
            
        Returns:
            Fusion strategy instance
        """
        strategy_class = self.get_strategy(name, version)
        return strategy_class(name=name, version=version or "1.0.0", **kwargs)
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """
        List all versions for a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            List of version strings
        """
        if name not in self._strategies:
            raise FusionStrategyNotFoundError(
                f"Fusion strategy '{name}' not found",
                details={"name": name}
            )
        return list(self._strategies[name].keys())
    
    def get_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a strategy.
        
        Args:
            name: Strategy name
            version: Strategy version (uses latest if None)
            
        Returns:
            Metadata dictionary
        """
        if name not in self._metadata:
            raise FusionStrategyNotFoundError(
                f"Fusion strategy '{name}' not found",
                details={"name": name}
            )
        
        versions = self._metadata[name]
        
        if version:
            if version not in versions:
                raise FusionStrategyNotFoundError(
                    f"Fusion strategy '{name}' version '{version}' not found",
                    details={"name": name, "version": version}
                )
            return versions[version]
        else:
            # Return latest version metadata
            latest_version = max(versions.keys(), key=lambda v: self._version_key(v))
            return versions[latest_version]
    
    def discover_strategies(
        self,
        package_path: str,
        strategy_base_class: Type[FusionStrategy] = FusionStrategy
    ) -> int:
        """
        Discover and register strategies from a package.
        
        Args:
            package_path: Python package path (e.g., 'chartradar.fusion.strategies')
            strategy_base_class: Base class to search for
            
        Returns:
            Number of strategies discovered and registered
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
                        
                        # Find strategy classes
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, strategy_base_class) and
                                obj is not strategy_base_class):
                                try:
                                    self.register(obj)
                                    count += 1
                                except Exception:
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
        Unregister a fusion strategy.
        
        Args:
            name: Strategy name
            version: Strategy version (removes all versions if None)
        """
        if name not in self._strategies:
            return
        
        if version:
            if version in self._strategies[name]:
                del self._strategies[name][version]
                if version in self._metadata[name]:
                    del self._metadata[name][version]
                
                # Clean up if no versions left
                if not self._strategies[name]:
                    del self._strategies[name]
                    del self._metadata[name]
        else:
            # Remove all versions
            del self._strategies[name]
            del self._metadata[name]


# Global registry instance
_default_registry = FusionStrategyRegistry()


def register_fusion_strategy(
    name: Optional[str] = None,
    version: Optional[str] = None,
    registry: Optional[FusionStrategyRegistry] = None
):
    """
    Decorator to register a fusion strategy class.
    
    Usage:
        @register_fusion_strategy(name="my_strategy", version="1.0.0")
        class MyFusionStrategy(FusionStrategy):
            ...
    
    Args:
        name: Strategy name (uses class name if None)
        version: Strategy version
        registry: Registry to use (uses default if None)
    """
    def decorator(cls: Type[FusionStrategy]):
        reg = registry or _default_registry
        reg.register(cls, name=name, version=version)
        return cls
    return decorator


def get_fusion_strategy(name: str, version: Optional[str] = None) -> Type[FusionStrategy]:
    """
    Get a fusion strategy class from the default registry.
    
    Args:
        name: Strategy name
        version: Strategy version (uses latest if None)
        
    Returns:
        Fusion strategy class
    """
    return _default_registry.get_strategy(name, version)


def create_fusion_strategy(name: str, version: Optional[str] = None, **kwargs: Any) -> FusionStrategy:
    """
    Create a fusion strategy instance from the default registry.
    
    Args:
        name: Strategy name
        version: Strategy version (uses latest if None)
        **kwargs: Parameters to pass to strategy constructor
        
    Returns:
        Fusion strategy instance
    """
    return _default_registry.create_strategy(name, version, **kwargs)

