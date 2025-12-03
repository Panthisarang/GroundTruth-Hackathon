import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple

import requests
from pymongo import MongoClient

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, 
    LLM_MAX_TOKENS, LLM_TEMPERATURE, DEBUG
)
from db import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class RateLimitExceeded(LLMError):
    """Raised when the rate limit is exceeded."""
    pass

class LLMClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self):
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL
        self.model = LLM_MODEL
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour TTL for cache
        
        # Rate limiting
        self.last_call_time = 0
        self.min_interval = 0.1  # 100ms between calls to avoid rate limiting
    
    def _get_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Generate a cache key for the given messages and parameters."""
        cache_dict = {
            "messages": messages,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **{k: v for k, v in kwargs.items() if k not in ['stream', 'n']}
        }
        return hashlib.md5(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if the response is in the cache and still valid."""
        cached = db.embeddings.find_one({
            "cache_key": cache_key,
            "created_at": {"$gt": time.time() - self.cache_ttl}
        })
        return cached.get("response") if cached else None
    
    def _save_to_cache(self, cache_key: str, response: str) -> None:
        """Save the response to the cache."""
        db.embeddings.update_one(
            {"cache_key": cache_key},
            {
                "$set": {
                    "response": response,
                    "created_at": time.time()
                }
            },
            upsert=True
        )
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()
    
    def _call_api(self, messages: List[Dict], **kwargs) -> Dict:
        """Make the actual API call to the LLM."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs
        }
        
        try:
            self._rate_limit()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                time.sleep(retry_after)
                return self._call_api(messages, **kwargs)
            logger.error(f"HTTP error calling LLM API: {e}")
            raise LLMError(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            raise LLMError(f"Failed to connect to LLM service: {e}")
    
    def call_llm(
        self,
        messages: List[Dict],
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """Call the LLM with the given messages and return the response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend to messages
            use_cache: Whether to use cached responses
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The generated text response
            
        Raises:
            LLMError: If the API call fails
            RateLimitExceeded: If rate limited and retries are exhausted
        """
        if not self.api_key:
            raise LLMError("API key not configured")
        
        # Prepend system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Check cache
        cache_key = self._get_cache_key(messages, **kwargs) if use_cache else None
        if use_cache and cache_key:
            cached_response = self._check_cache(cache_key)
            if cached_response:
                logger.debug("Cache hit for LLM call")
                return cached_response
        
        try:
            # Make the API call
            response = self._call_api(messages, **kwargs)
            content = response['choices'][0]['message']['content']
            
            # Cache the response
            if use_cache and cache_key:
                self._save_to_cache(cache_key, content)
            
            return content
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            raise LLMError(f"Failed to get response from LLM: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            return []
            
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached = db.embeddings.find_one({"cache_key": cache_key})
        
        if cached:
            return cached["embedding"]
        
        try:
            self._rate_limit()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "text-embedding-ada-002",  # Or your preferred embedding model
                "input": text
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            embedding = response.json()["data"][0]["embedding"]
            
            # Cache the embedding
            db.embeddings.insert_one({
                "cache_key": cache_key,
                "text": text,
                "embedding": embedding,
                "created_at": time.time()
            })
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default dimension for text-embedding-ada-002

# Create a singleton instance
llm_client = LLMClient()
