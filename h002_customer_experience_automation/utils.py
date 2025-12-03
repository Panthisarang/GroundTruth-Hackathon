import os
import json
import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps
import math

import numpy as np
from geopy.distance import great_circle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float, 
    unit: str = 'km'
) -> float:
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2
        unit: Unit of distance ('km' for kilometers, 'mi' for miles)
        
    Returns:
        float: Distance between the two points in the specified unit
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371.0
    
    # Calculate the distance
    distance = c * r
    
    # Convert to miles if requested
    if unit.lower() == 'mi':
        distance *= 0.621371
    
    return distance

def format_timedelta(delta: timedelta) -> str:
    """
    Format a timedelta into a human-readable string.
    
    Args:
        delta: The timedelta to format
        
    Returns:
        str: Formatted string (e.g., "2 hours, 30 minutes")
    """
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0 and not parts:  # Only show seconds if no hours or minutes
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
    
    return ", ".join(parts) if parts else "less than a minute"

def get_nearest_stores(
    lat: float, 
    lon: float, 
    limit: int = 5, 
    max_distance_km: float = 10.0,
    location_types: List[str] = None,
    user_preferences: Dict = None
) -> List[Dict[str, Any]]:
    """
    Find the nearest locations to the given coordinates.
    
    Args:
        lat: Latitude of the reference point
        lon: Longitude of the reference point
        limit: Maximum number of locations to return
        max_distance_km: Maximum distance in kilometers
        location_types: Optional list of location types to filter by
        user_preferences: Optional user preferences for relevance scoring
        
    Returns:
        List of location documents with distance and relevance information
    """
    from db import db
    
    try:
        # Use the get_nearby_locations method from the Database class
        locations = db.get_nearby_locations(
            coordinates=[lon, lat],  # [longitude, latitude]
            max_distance_km=max_distance_km,
            limit=limit,
            location_types=location_types,
            user_preferences=user_preferences
        )
        
        return locations
        
    except Exception as e:
        logger.error(f"Error finding nearby locations: {e}")
        return []

def get_relevant_coupons(
    user_id: str, 
    location_ids: List[str] = None, 
    limit: int = 5,
    user_preferences: Dict = None
) -> List[Dict[str, Any]]:
    """
    Get relevant coupons for the user, optionally filtered by location.
    
    Args:
        user_id: ID of the user
        location_ids: Optional list of location IDs to filter by
        limit: Maximum number of coupons to return
        user_preferences: Optional user preferences for relevance scoring
        
    Returns:
        List of coupon documents with relevance scores
    """
    from db import db
    from datetime import datetime
    
    try:
        # Build the base query
        query = {
            "valid_until": {"$gte": datetime.utcnow()},
            "is_active": True
        }
        
        if location_ids:
            query["location_id"] = {"$in": location_ids}
        
        # Get active coupons with location details
        pipeline = [
            {"$match": query},
            {
                "$lookup": {
                    "from": "locations",
                    "localField": "location_id",
                    "foreignField": "_id",
                    "as": "location"
                }
            },
            {"$unwind": "$location"},
            {"$limit": limit * 3}  # Get more than needed for sorting
        ]
        
        coupons = list(db.coupons.aggregate(pipeline))
        
        if not coupons:
            return []
        
        # Get user's order history for personalization
        user_orders = list(db.orders.find(
            {"user_id": user_id},
            {"location_id": 1, "items": 1, "created_at": 1}
        ).sort("created_at", -1).limit(50))  # Get recent orders
        
        # Get user's favorite locations if available
        user = db.users.find_one({"_id": user_id}, {"preferences.favorite_locations": 1})
        favorite_locations = user.get("preferences", {}).get("favorite_locations", []) if user else []
        
        # Enhanced scoring function for coupon relevance
        def score_coupon(coupon: Dict[str, Any]) -> float:
            score = 0.0
            location = coupon.get("location", {})
            
            # Base score based on coupon type and value
            if coupon["discount_type"] == "PERCENT_OFF":
                score += min(coupon["discount_value"] / 10, 5)  # Max 5 points for 50%+ off
            else:
                # Flat discount - normalize based on order value (assuming average order value)
                score += min(coupon["discount_value"] / 10, 3)  # Max 3 points for $30+ off
            
            # Time-based scoring (higher for coupons expiring soon)
            time_left = (coupon["valid_until"] - datetime.utcnow()).total_seconds()
            if time_left < 24 * 3600:  # 1 day
                score += 3.0
            elif time_left < 3 * 24 * 3600:  # 3 days
                score += 1.5
            elif time_left < 7 * 24 * 3600:  # 1 week
                score += 0.5
            
            # Location-based scoring
            location_id = str(coupon.get("location_id", ""))
            
            # Higher score for favorite locations
            if location_id in favorite_locations:
                score += 2.0
                
            # Higher score for locations the user has visited before
            order_count = sum(1 for order in user_orders if order.get("location_id") == location_id)
            score += min(order_count * 0.5, 3)  # Max 3 points for frequent visits
            
            # Preference matching (if user preferences are provided)
            if user_preferences and location:
                # Match location type preferences
                location_type = location.get("type")
                if location_type in user_preferences.get("preferred_location_types", []):
                    score += 1.0
                
                # Match cuisine preferences for restaurants/cafes
                if location_type in ["restaurant", "cafe"]:
                    location_cuisines = set(location.get("cuisine", []))
                    preferred_cuisines = set(user_preferences.get("preferred_cuisines", []))
                    if location_cuisines.intersection(preferred_cuisines):
                        score += 1.5
            
            return score
        
        # Score and sort coupons
        for coupon in coupons:
            coupon["relevance_score"] = score_coupon(coupon)
        
        # Sort by score (descending) and limit results
        coupons.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return coupons[:limit]
        
    except Exception as e:
        logger.error(f"Error getting relevant coupons: {e}")
        return []

def mask_sensitive_data(data: Any) -> Any:
    """
    Recursively mask sensitive data in a dictionary or list.
    
    Args:
        data: The data to mask (dict, list, or primitive type)
        
    Returns:
        A copy of the data with sensitive fields masked
    """
    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            # Mask common sensitive fields
            if isinstance(key, str) and any(s in key.lower() for s in [
                'password', 'secret', 'token', 'key', 'credit', 'card', 'cvv',
                'ssn', 'social', 'security', 'dob', 'birth', 'phone', 'email'
            ]):
                masked[key] = '***MASKED***'
            else:
                masked[key] = mask_sensitive_data(value)
        return masked
    elif isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    else:
        return data

def log_api_call(func: Callable) -> Callable:
    """
    Decorator to log API calls with their inputs and outputs.
    
    Logs the function name, arguments, return value, and execution time.
    Masks sensitive data in the logs.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Format function call for logging
        func_name = func.__name__
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        logger.info(f"Calling {func_name}({signature})")
        
        # Time the function execution
        start_time = time.time()
        
        try:
            # Call the function
            result = func(*args, **kwargs)
            
            # Log the result (masking sensitive data)
            exec_time = time.time() - start_time
            logger.info(
                f"{func_name} returned in {exec_time:.3f}s: "
                f"{mask_sensitive_data(result)}"
            )
            
            return result
            
        except Exception as e:
            # Log the exception
            exec_time = time.time() - start_time
            logger.error(
                f"{func_name} failed after {exec_time:.3f}s: {str(e)}",
                exc_info=True
            )
            raise
    
    return wrapper

def generate_id(prefix: str = '') -> str:
    """
    Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        A unique string ID
    """
    import uuid
    return f"{prefix}{uuid.uuid4().hex}"

def parse_bool(value: Any) -> bool:
    """
    Parse a value to a boolean.
    
    Args:
        value: The value to parse
        
    Returns:
        The parsed boolean value
        
    Raises:
        ValueError: If the value cannot be parsed to a boolean
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ('true', 't', 'yes', 'y', '1'):
            return True
        if value.lower() in ('false', 'f', 'no', 'n', '0'):
            return False
    raise ValueError(f"Cannot convert {value} to boolean")

def validate_email(email: str) -> bool:
    """
    Validate an email address format.
    
    Args:
        email: The email address to validate
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of the specified size.
    
    Args:
        lst: The list to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunks (lists)
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

class Timer:
    """Context manager for timing code blocks."""
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000
