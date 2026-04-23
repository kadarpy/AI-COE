"""
Deterministic Cache Manager for RAG Evaluation
==============================================

Ensures reproducibility by caching:
- RAG responses (question + context + model) → response
- Evaluation LLM calls (prompt + model) → evaluation result

Uses SHA256 hashing for cache keys.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DeterministicCacheManager:
    """
    Deterministic caching system for RAG evaluation.
    
    Cache behavior:
    - If cache exists → reuse response
    - If not → compute + store
    - All keys are deterministic (SHA256)
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache manager initialized at {self.cache_dir}")

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of data for cache key.
        
        Args:
            data: Dictionary to hash
            
        Returns:
            SHA256 hex digest
        """
        # Sort keys for deterministic hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_cache_path(self, hash_key: str) -> Path:
        """Get cache file path for given hash key."""
        return self.cache_dir / f"{hash_key}.json"

    def get(self, key_data: Dict[str, Any], cache_type: str = "rag") -> Optional[Dict[str, Any]]:
        """
        Get value from cache if exists.
        
        Args:
            key_data: Dictionary containing cache key components
            cache_type: Type of cache (rag, evaluation, etc)
            
        Returns:
            Cached value dict, or None if not found
        """
        # Add cache type to key for namespace separation
        full_key = {"type": cache_type, **key_data}
        hash_key = self._compute_hash(full_key)
        cache_path = self._get_cache_path(hash_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {hash_key[:8]}... ({cache_type})")
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            logger.debug(f"Cache hit: {hash_key[:8]}... ({cache_type})")
            return cached.get("value")
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_path}: {e}")
            return None

    def set(self, key_data: Dict[str, Any], value: Any, cache_type: str = "rag") -> bool:
        """
        Store value in cache.
        
        Args:
            key_data: Dictionary containing cache key components
            value: Value to cache
            cache_type: Type of cache (rag, evaluation, etc)
            
        Returns:
            True if successful, False otherwise
        """
        full_key = {"type": cache_type, **key_data}
        hash_key = self._compute_hash(full_key)
        cache_path = self._get_cache_path(hash_key)

        try:
            cache_entry = {
                "hash_key": hash_key,
                "type": cache_type,
                "key_data": key_data,
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_entry, f, indent=2, default=str)
            logger.debug(f"Cache write: {hash_key[:8]}... ({cache_type})")
            return True
        except Exception as e:
            logger.error(f"Failed to write cache {cache_path}: {e}")
            return False

    def clear(self, cache_type: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            cache_type: Clear only this type, or None to clear all
            
        Returns:
            Number of files deleted
        """
        deleted = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_type is None:
                cache_file.unlink()
                deleted += 1
            else:
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                    if entry.get("type") == cache_type:
                        cache_file.unlink()
                        deleted += 1
                except:
                    pass
        logger.info(f"Cleared {deleted} cache entries (type={cache_type})")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            "total_entries": 0,
            "by_type": {},
            "total_size_bytes": 0,
            "cache_dir": str(self.cache_dir)
        }

        for cache_file in self.cache_dir.glob("*.json"):
            stats["total_entries"] += 1
            stats["total_size_bytes"] += cache_file.stat().st_size

            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                cache_type = entry.get("type", "unknown")
                stats["by_type"][cache_type] = stats["by_type"].get(cache_type, 0) + 1
            except:
                pass

        return stats


# Global cache instance
_cache_manager = None


def get_cache_manager(cache_dir: str = ".cache") -> DeterministicCacheManager:
    """
    Get or create global cache manager instance.
    
    Args:
        cache_dir: Cache directory path
        
    Returns:
        DeterministicCacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DeterministicCacheManager(cache_dir)
    return _cache_manager


def cache_rag_response(question: str, context: str, model_id: str, response: str) -> bool:
    """
    Cache a RAG response.
    
    Args:
        question: The question asked
        context: The context used
        model_id: The model ID used
        response: The model response
        
    Returns:
        True if cached successfully
    """
    cache = get_cache_manager()
    key_data = {
        "question": question,
        "context": context,
        "model_id": model_id
    }
    return cache.set(key_data, {"response": response}, cache_type="rag")


def get_cached_rag_response(question: str, context: str, model_id: str) -> Optional[str]:
    """
    Get cached RAG response.
    
    Args:
        question: The question asked
        context: The context used
        model_id: The model ID used
        
    Returns:
        Cached response, or None if not found
    """
    cache = get_cache_manager()
    key_data = {
        "question": question,
        "context": context,
        "model_id": model_id
    }
    cached = cache.get(key_data, cache_type="rag")
    if cached:
        return cached.get("response")
    return None


def cache_evaluation_call(prompt: str, model_id: str, result: str) -> bool:
    """
    Cache an evaluation LLM call.
    
    Args:
        prompt: The prompt sent to LLM
        model_id: The model ID used
        result: The LLM result
        
    Returns:
        True if cached successfully
    """
    cache = get_cache_manager()
    key_data = {
        "prompt": prompt,
        "model_id": model_id
    }
    return cache.set(key_data, {"result": result}, cache_type="evaluation")


def get_cached_evaluation_call(prompt: str, model_id: str) -> Optional[str]:
    """
    Get cached evaluation LLM call result.
    
    Args:
        prompt: The prompt sent to LLM
        model_id: The model ID used
        
    Returns:
        Cached result, or None if not found
    """
    cache = get_cache_manager()
    key_data = {
        "prompt": prompt,
        "model_id": model_id
    }
    cached = cache.get(key_data, cache_type="evaluation")
    if cached:
        return cached.get("result")
    return None
