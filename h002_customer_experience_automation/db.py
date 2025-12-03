import pymongo
from pymongo import MongoClient
from bson import ObjectId
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from h002_customer_experience_automation.config import (
    MONGODB_URI, DB_NAME, USERS_COLLECTION, ORDERS_COLLECTION,
    STORES_COLLECTION, COUPONS_COLLECTION, CHAT_HISTORY_COLLECTION,
    EMBEDDINGS_COLLECTION
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the database connection and collections."""
        if self._initialized:
            return
            
        try:
            # Connect to MongoDB
            self.client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,        # 10 second connection timeout
                socketTimeoutMS=45000,          # 45 second timeout for operations
                retryWrites=True,
                retryReads=True
            )
            
            # Test the connection
            self.client.admin.command('ping')
            
            self.db = self.client[DB_NAME]
            
            # Initialize collections
            self.users = self.db[USERS_COLLECTION]
            self.orders = self.db[ORDERS_COLLECTION]
            self.locations = self.db[STORES_COLLECTION]  # Renamed from stores to locations
            self.coupons = self.db[COUPONS_COLLECTION]
            self.chat_history = self.db[CHAT_HISTORY_COLLECTION]
            self.embeddings = self.db[EMBEDDINGS_COLLECTION]
            
            # Create indexes
            self._create_indexes()
            
            # Mark as initialized
            self._initialized = True
            
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create necessary indexes for better query performance."""
        try:
            # Users collection
            self.users.create_index("email", unique=True)
            
            # Orders collection
            self.orders.create_index("user_id")
            self.orders.create_index("location_id")
            
            # Locations collection
            try:
                # Check if the collection exists and has documents
                if self.locations.count_documents({}) > 0:
                    # If collection has data, create index in background
                    self.locations.create_index([("location", "2dsphere")], background=True)
                else:
                    # If collection is empty, create index normally
                    self.locations.create_index([("location", "2dsphere")])
                
                self.locations.create_index("type")  # For filtering by location type
            except Exception as e:
                logger.warning(f"Could not create 2dsphere index: {e}")
                # Fallback to regular index
                self.locations.create_index("location")
            
            # Coupons collection
            self.coupons.create_index("location_id")
            
            # Chat history collection
            self.chat_history.create_index("user_id")
            
            # Embeddings collection
            self.embeddings.create_index("content_type")
            self.embeddings.create_index("content_id")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database indexes: {e}")
            # Don't raise the exception, as the app might still work without some indexes
        
    def _initialize_default_data(self):
        """Initialize default location types and preferences if they don't exist."""
        # Default location types
        location_types = [
            "restaurant", "cafe", "park", "library", "gym", 
            "mall", "cinema", "museum", "hotel", "hospital"
        ]
        
        # Default preferences
        preference_categories = [
            "cuisine", "price_range", "accessibility", 
            "atmosphere", "amenities"
        ]
        
        # Store in a settings collection
        settings = self.db.settings
        
        if not settings.find_one({"key": "location_types"}):
            settings.insert_one({
                "key": "location_types",
                "value": location_types
            })
            
        if not settings.find_one({"key": "preference_categories"}):
            settings.insert_one({
                "key": "preference_categories",
                "value": preference_categories
            })
    
    # User Operations
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Retrieve a user by email."""
        return self.users.find_one({"email": email})
    
    def create_user(self, user_data: Dict) -> str:
        """Create a new user."""
        result = self.users.insert_one(user_data)
        return str(result.inserted_id)
    
    def update_user_profile(self, user_id: str, update_data: Dict) -> bool:
        """Update user profile data."""
        result = self.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Retrieve a user by ID with their preferences."""
        if not isinstance(user_id, str):
            user_id = str(user_id)
        
        # Get user with default preferences if they don't exist
        user = self.users.find_one({"_id": ObjectId(user_id)})
        
        if user:
            # Ensure default preferences exist
            if 'preferences' not in user:
                user['preferences'] = {
                    'cuisine': [],
                    'price_range': 'medium',
                    'accessibility': [],
                    'atmosphere': [],
                    'amenities': []
                }
                self.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"preferences": user['preferences']}}
                )
            
            # Ensure location preferences exist
            if 'location' not in user:
                user['location'] = {
                    'type': 'Point',
                    'coordinates': [0, 0],  # Default to null island
                    'address': 'Unknown location',
                    'last_updated': datetime.utcnow()
                }
                self.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"location": user['location']}}
                )
                
            user["_id"] = str(user["_id"])
        return user
    
    # Chat Operations
    def store_chat_message(self, user_id: str, role: str, content: str, metadata: Dict = None) -> str:
        """Store a chat message in the database."""
        message = {
            "user_id": ObjectId(user_id),
            "role": role,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        result = self.chat_history.insert_one(message)
        return str(result.inserted_id)
    
    def get_recent_chat_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent chat history for a user."""
        cursor = self.chat_history.find(
            {"user_id": ObjectId(user_id)}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    # LLM Usage Tracking
    def increment_llm_usage(self, user_id: str) -> int:
        """Increment and return the daily LLM usage count for a user."""
        today = datetime.utcnow().date()
        result = self.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$inc": {"llm_usage_count": 1},
                "$setOnInsert": {"last_llm_usage_date": today}
            },
            upsert=True
        )
        return self.get_today_llm_usage(user_id)
    
    def get_today_llm_usage(self, user_id: str) -> int:
        """Get today's LLM usage count for a user."""
        user = self.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return 0
        
        today = datetime.utcnow().date()
        last_usage_date = user.get("last_llm_usage_date")
        
        if last_usage_date and isinstance(last_usage_date, datetime):
            last_usage_date = last_usage_date.date()
        
        if last_usage_date != today:
            return 0
        
        return user.get("llm_usage_count", 0)
    
    # Store and Coupon Operations
    def get_nearby_locations(
        self, 
        coordinates: List[float], 
        location_types: List[str] = None,
        user_preferences: Dict = None,
        max_distance_km: float = 5, 
        limit: int = 10
    ) -> List[Dict]:
        """
        Find locations near the given coordinates, optionally filtered by type and preferences.
        
        Args:
            coordinates: [longitude, latitude]
            location_types: List of location types to filter by (e.g., ['restaurant', 'cafe'])
            user_preferences: Dictionary of user preferences for filtering
            max_distance_km: Maximum distance in kilometers
            limit: Maximum number of results to return
            
        Returns:
            List of nearby locations with distance and relevance information
        """
        try:
            max_distance_meters = max_distance_km * 1000
            
            # Build the base query
            query = {
                "location": {
                    "$near": {
                        "$geometry": {
                            "type": "Point",
                            "coordinates": coordinates
                        },
                        "$maxDistance": max_distance_meters
                    }
                }
            }
            
            # Add location type filter if specified
            if location_types and len(location_types) > 0:
                query["type"] = {"$in": location_types}
            
            # Add preference-based filters if provided
            if user_preferences:
                if 'price_range' in user_preferences and user_preferences['price_range']:
                    query["price_level"] = {"$lte": len(user_preferences['price_range']) + 1}
                
                # Add more preference-based filters as needed
                # Example: cuisine, accessibility features, etc.
            
            # Execute the query with pagination
            locations = list(self.locations.find(query).limit(limit))
            
            # Calculate distances and format results
            results = []
            for loc in locations:
                # Calculate distance
                distance_km = self._calculate_distance(
                    coordinates[1], coordinates[0],  # lat, lon
                    loc['location']['coordinates'][1], 
                    loc['location']['coordinates'][0]
                )
                
                # Calculate relevance score based on distance and preferences
                relevance = self._calculate_relevance(loc, user_preferences, distance_km)
                
                # Format the result
                result = {
                    "_id": str(loc["_id"]),
                    "name": loc.get("name", "Unnamed Location"),
                    "type": loc.get("type", "unknown"),
                    "address": loc.get("address", ""),
                    "distance_km": round(distance_km, 2),
                    "relevance_score": relevance,
                    "rating": loc.get("rating", 0),
                    "price_level": loc.get("price_level", 0),
                    "opening_hours": loc.get("opening_hours", {})
                }
                
                # Add any additional fields
                for field in ["description", "website", "phone", "photos"]:
                    if field in loc:
                        result[field] = loc[field]
                
                results.append(result)
            
            # Sort by relevance (descending) and then by distance (ascending)
            results.sort(key=lambda x: (-x["relevance_score"], x["distance_km"]))
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding nearby locations: {e}")
            return []
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in kilometers using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Radius of Earth in kilometers
        R = 6371.0
        
        return R * c
    
    def _calculate_relevance(self, location, user_preferences, distance_km):
        """Calculate a relevance score for a location based on user preferences and distance."""
        if not user_preferences:
            # If no preferences, base relevance on distance only
            return 1.0 / (1.0 + distance_km)
        
        score = 0.0
        
        # Base score from distance (closer is better)
        distance_score = 1.0 / (1.0 + distance_km)
        
        # Preference matching
        preference_score = 0.0
        
        # Example: Match cuisine preferences
        if 'cuisine' in user_preferences and 'cuisine' in location:
            matching_cuisines = set(user_preferences['cuisine']) & set(location['cuisine'])
            if matching_cuisines:
                preference_score += 0.3 * len(matching_cuisines)
        
        # Example: Price range matching
        if 'price_range' in user_preferences and 'price_level' in location:
            user_price_level = len(user_preferences['price_range'])  # low=1, medium=2, high=3
            if user_price_level >= location['price_level']:
                preference_score += 0.2
        
        # Example: Rating importance
        rating = location.get('rating', 0)
        rating_score = (rating / 5.0) * 0.2  # Normalize to 0-0.2 range
        
        # Combine scores (adjust weights as needed)
        score = 0.5 * distance_score + 0.3 * preference_score + 0.2 * rating_score
        
        return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
    
    # Embedding Operations
    def store_embedding(self, doc_id: str, embedding: List[float], metadata: Dict) -> str:
        """Store a document embedding with metadata."""
        doc = {
            "doc_id": doc_id,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.utcnow()
        }
        result = self.embeddings.insert_one(doc)
        return str(result.inserted_id)
    
    def search_similar_embeddings(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Find the most similar documents to the query embedding."""
        # MongoDB Atlas supports vector search with $vectorSearch
        # For local development, we'll use a simple cosine similarity
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "doc_id": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            return list(self.embeddings.aggregate(pipeline))
        except:
            # Fallback for local development without vector search
            all_docs = list(self.embeddings.find({}))
            return sorted(
                all_docs,
                key=lambda x: self._cosine_similarity(query_embedding, x["embedding"]),
                reverse=True
            )[:top_k]
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)

# Global database instance
db = Database()
