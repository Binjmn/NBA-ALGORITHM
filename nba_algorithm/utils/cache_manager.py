#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Cache Management Module

This module provides a sophisticated caching system with the following features:
- Tiered caching strategy with different TTLs for different data types
- Cache metadata tracking for data freshness
- Automatic cache invalidation based on configurable thresholds
- Selective caching for different types of data
- Support for force-refresh to bypass cache when needed
"""

import os
import json
import time
import logging
import hashlib
from enum import Enum
from typing import Any, Dict, Optional, Union, List, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path('data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache subdirectories for different tiers
VOLATILE_CACHE_DIR = CACHE_DIR / 'volatile'  # Short-lived cache (minutes)
REGULAR_CACHE_DIR = CACHE_DIR / 'regular'    # Medium-lived cache (hours)
STABLE_CACHE_DIR = CACHE_DIR / 'stable'      # Long-lived cache (days/weeks)

# Create cache subdirectories
VOLATILE_CACHE_DIR.mkdir(exist_ok=True)
REGULAR_CACHE_DIR.mkdir(exist_ok=True)
STABLE_CACHE_DIR.mkdir(exist_ok=True)


class CacheTier(Enum):
    """Cache tier enumeration with associated TTLs"""
    VOLATILE = 1  # Minutes (player status, injury updates, lineups)
    REGULAR = 2   # Hours (team stats, recent performances)
    STABLE = 3    # Days/Weeks (historical data, team info)


# Default TTLs for each cache tier in seconds
DEFAULT_TTL = {
    CacheTier.VOLATILE: 60 * 15,       # 15 minutes
    CacheTier.REGULAR: 60 * 60 * 6,     # 6 hours
    CacheTier.STABLE: 60 * 60 * 24 * 7   # 7 days
}

# Cache configuration by data type
CACHE_CONFIG = {
    # Volatile data (short TTL)
    'player_status': {'tier': CacheTier.VOLATILE, 'ttl': 60 * 5},        # 5 minutes
    'injuries': {'tier': CacheTier.VOLATILE, 'ttl': 60 * 10},            # 10 minutes
    'lineups': {'tier': CacheTier.VOLATILE, 'ttl': 60 * 15},             # 15 minutes
    'live_odds': {'tier': CacheTier.VOLATILE, 'ttl': 60 * 5},            # 5 minutes
    
    # Regular data (medium TTL)
    'team_stats': {'tier': CacheTier.REGULAR, 'ttl': 60 * 60 * 3},       # 3 hours
    'player_stats': {'tier': CacheTier.REGULAR, 'ttl': 60 * 60 * 4},     # 4 hours
    'upcoming_games': {'tier': CacheTier.REGULAR, 'ttl': 60 * 60 * 2},   # 2 hours
    'odds': {'tier': CacheTier.REGULAR, 'ttl': 60 * 60 * 1},             # 1 hour
    
    # Stable data (long TTL)
    'teams': {'tier': CacheTier.STABLE, 'ttl': 60 * 60 * 24 * 14},       # 14 days
    'players': {'tier': CacheTier.STABLE, 'ttl': 60 * 60 * 24 * 7},      # 7 days
    'historical_games': {'tier': CacheTier.STABLE, 'ttl': 60 * 60 * 24 * 30}, # 30 days
    'historical_stats': {'tier': CacheTier.STABLE, 'ttl': 60 * 60 * 24 * 30}, # 30 days
}


def get_cache_path(data_type: str, identifier: Optional[str] = None) -> Path:
    """Get the path for a cache file based on its data type and optional identifier"""
    # Determine which cache directory to use based on data type
    config = CACHE_CONFIG.get(data_type, {'tier': CacheTier.REGULAR})
    tier = config['tier']
    
    if tier == CacheTier.VOLATILE:
        cache_dir = VOLATILE_CACHE_DIR
    elif tier == CacheTier.STABLE:
        cache_dir = STABLE_CACHE_DIR
    else:  # Default to REGULAR tier
        cache_dir = REGULAR_CACHE_DIR
    
    # Create filename with optional identifier
    if identifier:
        filename = f"{data_type}_{identifier}.json"
    else:
        filename = f"{data_type}.json"
    
    return cache_dir / filename


def generate_cache_key(params: Dict) -> str:
    """Generate a unique cache key based on request parameters"""
    # Sort the parameters to ensure consistent keys
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


def read_cache(data_type: str, identifier: Optional[str] = None, 
              force_refresh: bool = False) -> Optional[Dict]:
    """Read data from cache if it exists, is not expired, and force_refresh is False
    
    Args:
        data_type: Type of data (used to determine appropriate TTL)
        identifier: Optional identifier for the cached data
        force_refresh: If True, bypass cache and return None
        
    Returns:
        Cached data as dict or None if cache is invalid or force_refresh is True
    """
    if force_refresh:
        logger.debug(f"Force refresh requested for {data_type}, bypassing cache")
        return None
    
    cache_path = get_cache_path(data_type, identifier)
    
    try:
        if not cache_path.exists():
            return None
        
        # Get TTL for this data type
        config = CACHE_CONFIG.get(data_type, {'ttl': DEFAULT_TTL[CacheTier.REGULAR]})
        ttl = config.get('ttl', DEFAULT_TTL[CacheTier.REGULAR])
        
        # Read the cache file
        with cache_path.open('r') as f:
            cache_data = json.load(f)
        
        # Check if cache contains metadata
        if not isinstance(cache_data, dict) or 'timestamp' not in cache_data:
            logger.warning(f"Cache file {cache_path} is missing metadata, invalidating")
            return None
        
        # Check if cache is expired
        cache_time = cache_data['timestamp']
        current_time = time.time()
        
        if current_time - cache_time > ttl:
            logger.debug(f"Cache expired for {data_type} (age: {(current_time - cache_time)/60:.1f} minutes)")
            return None
        
        # Return the actual data
        return cache_data['data']
    
    except Exception as e:
        logger.warning(f"Error reading cache {cache_path}: {str(e)}")
        return None


def write_cache(data_type: str, data: Any, identifier: Optional[str] = None, 
               metadata: Optional[Dict] = None) -> bool:
    """Write data to cache with metadata
    
    Args:
        data_type: Type of data (used to determine cache location)
        data: The data to cache
        identifier: Optional identifier for the cached data
        metadata: Optional additional metadata to store with the cache
        
    Returns:
        Boolean indicating success
    """
    cache_path = get_cache_path(data_type, identifier)
    
    try:
        # Prepare cache entry with metadata
        cache_entry = {
            'timestamp': time.time(),
            'data': data,
            'data_type': data_type,
            'cache_version': '1.0'
        }
        
        # Add any custom metadata
        if metadata:
            cache_entry['metadata'] = metadata
        
        # Write to cache file
        with cache_path.open('w') as f:
            json.dump(cache_entry, f)
        
        logger.debug(f"Cached {data_type} data at {cache_path}")
        return True
    
    except Exception as e:
        logger.warning(f"Error writing cache {cache_path}: {str(e)}")
        return False


def clear_cache(data_type: Optional[str] = None, 
                tier: Optional[CacheTier] = None,
                older_than: Optional[int] = None) -> int:
    """Clear cache files based on type, tier, or age
    
    Args:
        data_type: Specific data type to clear (optional)
        tier: Specific cache tier to clear (optional)
        older_than: Clear caches older than this many seconds (optional)
        
    Returns:
        Number of cache files cleared
    """
    cleared_count = 0
    
    try:
        # Determine which directories to search
        if tier == CacheTier.VOLATILE:
            search_dirs = [VOLATILE_CACHE_DIR]
        elif tier == CacheTier.REGULAR:
            search_dirs = [REGULAR_CACHE_DIR]
        elif tier == CacheTier.STABLE:
            search_dirs = [STABLE_CACHE_DIR]
        else:
            search_dirs = [VOLATILE_CACHE_DIR, REGULAR_CACHE_DIR, STABLE_CACHE_DIR]
        
        # Find and clear matching cache files
        for cache_dir in search_dirs:
            if not cache_dir.exists():
                continue
                
            for cache_file in cache_dir.glob(f"{'*' if data_type is None else data_type+'*'}.json"):
                should_delete = True
                
                # Check file age if specified
                if older_than is not None:
                    file_time = cache_file.stat().st_mtime
                    if time.time() - file_time <= older_than:
                        should_delete = False
                
                if should_delete:
                    cache_file.unlink()
                    cleared_count += 1
                    logger.debug(f"Cleared cache file: {cache_file}")
        
        return cleared_count
    
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return cleared_count


class CacheManager:
    """Class-based interface for cache management
    
    This class provides an object-oriented interface to the cache system,
    allowing for specialized caching behavior for different components.
    Each instance can have its own cache name and TTL settings.
    """
    
    def __init__(self, cache_name: str, ttl_seconds: int = 3600, tier: CacheTier = CacheTier.REGULAR):
        """Initialize a cache manager instance
        
        Args:
            cache_name: Name for this cache instance (used as identifier)
            ttl_seconds: Time-to-live in seconds for cached items
            tier: Cache tier to use (VOLATILE, REGULAR, or STABLE)
        """
        self.cache_name = cache_name
        self.ttl_seconds = ttl_seconds
        self.tier = tier
        
        # Ensure the cache type exists in the config, or add it
        if cache_name not in CACHE_CONFIG:
            CACHE_CONFIG[cache_name] = {
                'tier': tier,
                'ttl': ttl_seconds
            }
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value or None if not found or expired
        """
        identifier = f"{key}"
        return read_cache(self.cache_name, identifier)
    
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None) -> bool:
        """Set a value in cache
        
        Args:
            key: Cache key to store under
            value: Value to cache
            metadata: Optional metadata to store with the value
            
        Returns:
            Boolean indicating success
        """
        identifier = f"{key}"
        return write_cache(self.cache_name, value, identifier, metadata)
    
    def clear(self, older_than: Optional[int] = None) -> int:
        """Clear all cached items for this cache instance
        
        Args:
            older_than: Optional age threshold in seconds
            
        Returns:
            Number of cache items cleared
        """
        return clear_cache(self.cache_name, older_than=older_than)
    
    def refresh(self, key: str, getter_func, *args, **kwargs) -> Any:
        """Get a value from cache, or call a function to get and cache it
        
        Args:
            key: Cache key to retrieve
            getter_func: Function to call if cache miss
            *args, **kwargs: Arguments to pass to getter_func
            
        Returns:
            Cached or freshly retrieved value
        """
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Cache miss, call the getter function
        fresh_value = getter_func(*args, **kwargs)
        self.set(key, fresh_value)
        return fresh_value


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the current cache state
    
    Returns:
        Dictionary with cache statistics
    """
    stats = {
        'volatile_count': 0,
        'volatile_size_kb': 0,
        'regular_count': 0,
        'regular_size_kb': 0,
        'stable_count': 0,
        'stable_size_kb': 0,
        'total_count': 0,
        'total_size_kb': 0,
        'data_types': {}
    }
    
    try:
        # Collect stats from each cache directory
        for tier_name, cache_dir in [
            ('volatile', VOLATILE_CACHE_DIR),
            ('regular', REGULAR_CACHE_DIR),
            ('stable', STABLE_CACHE_DIR)
        ]:
            if not cache_dir.exists():
                continue
                
            for cache_file in cache_dir.glob('*.json'):
                # Extract data type from filename
                file_name = cache_file.name
                data_type = file_name.split('_')[0] if '_' in file_name else file_name.replace('.json', '')
                
                # Get file size
                size_kb = cache_file.stat().st_size / 1024
                
                # Update stats
                stats[f'{tier_name}_count'] += 1
                stats[f'{tier_name}_size_kb'] += size_kb
                stats['total_count'] += 1
                stats['total_size_kb'] += size_kb
                
                # Track by data type
                if data_type not in stats['data_types']:
                    stats['data_types'][data_type] = {
                        'count': 0,
                        'size_kb': 0
                    }
                
                stats['data_types'][data_type]['count'] += 1
                stats['data_types'][data_type]['size_kb'] += size_kb
        
        # Round sizes
        for key in stats:
            if key.endswith('_size_kb'):
                stats[key] = round(stats[key], 2)
        
        for data_type in stats['data_types']:
            stats['data_types'][data_type]['size_kb'] = round(stats['data_types'][data_type]['size_kb'], 2)
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return stats
