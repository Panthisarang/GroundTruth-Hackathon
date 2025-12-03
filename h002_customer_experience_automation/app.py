import os
import sys
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG', 'False').lower() == 'true' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
try:
    from h002_customer_experience_automation.config import (
        DEBUG, MAX_DAILY_LLM_CALLS, POLICIES_DIR, MONGODB_URI, DB_NAME
    )
    from h002_customer_experience_automation.db import Database
    from h002_customer_experience_automation.auth import auth_service
    from h002_customer_experience_automation.llm_client import llm_client
    from h002_customer_experience_automation.rag_pipeline import rag_pipeline
    from h002_customer_experience_automation.rate_limiter import rate_limiter
    from h002_customer_experience_automation.pii_masker import mask_pii, unmask_pii
    from h002_customer_experience_automation.utils import (
        get_nearest_stores, get_relevant_coupons, calculate_distance, format_timedelta
    )
    
    logger.info("All required modules imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Initialize database
try:
    logger.info("Initializing database connection...")
    db = Database()
    logger.info(f"Successfully connected to MongoDB at {MONGODB_URI}/{DB_NAME}")
    
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    st.error("Failed to connect to the database. Please check your connection and try again.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Customer Experience Automation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            padding: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #f0f2f6;
            margin-left: 20%;
        }
        .chat-message.assistant {
            background-color: #e3f2fd;
            margin-right: 20%;
        }
        .chat-message .avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
            flex-shrink: 0;
        }
        .chat-message .content {
            flex-grow: 1;
        }
        .store-card {
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .coupon-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #4a90e2;
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input {
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
    st.session_state.chat_history = []
    st.session_state.pending_message = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_login' not in st.session_state:
    st.session_state.show_login = True
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

def login_user(email: str, password: str) -> bool:
    """Authenticate a user."""
    try:
        user = auth_service.verify_user(email, password)
        if user:
            st.session_state.user = user
            st.session_state.show_login = False
            st.session_state.show_signup = False
            st.session_state.show_chat = True
            
            # Load user's chat history
            st.session_state.chat_history = list(db.chat_history.find(
                {"user_id": user["_id"]}
            ).sort("timestamp", -1).limit(20))
            
            st.rerun()
            return True
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def signup_user(user_data: Dict[str, Any]) -> bool:
    """Register a new user."""
    try:
        # Extract password and remove it from user_data
        password = user_data.pop('password')
        confirm_password = user_data.pop('confirm_password', '')
        
        if password != confirm_password:
            st.error("Passwords do not match")
            return False
            
        # Create the user
        user_id = auth_service.create_user(
            email=user_data['email'],
            password=password,
            **user_data
        )
        
        if user_id:
            st.success("Account created successfully! Please log in.")
            st.session_state.show_signup = False
            st.session_state.show_login = True
            st.rerun()
            return True
            
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return False

def logout_user():
    """Log out the current user."""
    st.session_state.user = None
    st.session_state.chat_history = []
    st.session_state.show_login = True
    st.session_state.show_chat = False
    st.rerun()

def process_user_message(user_input: str) -> str:
    """Process a user message and generate a response."""
    try:
        # Check rate limit
        rate_limiter.check_limit(st.session_state.user["_id"])
        
        # Mask PII before sending to LLM
        masked_input, pii_mapping = mask_pii(user_input)
        
        # Get user's location (simulated for demo)
        user_location = {
            "lat": 40.7128,  # Default to New York
            "lon": -74.0060,
            "city": "New York",
            "state": "NY"
        }
        
        # Get nearby stores
        nearby_stores = get_nearest_stores(
            user_location["lat"], 
            user_location["lon"],
            limit=3
        )
        
        # Get relevant coupons
        store_ids = [store["_id"] for store in nearby_stores]
        coupons = get_relevant_coupons(st.session_state.user["_id"], store_ids)
        
        # Format context for the LLM
        context = {
            "user": {
                "id": st.session_state.user["_id"],
                "name": f"{st.session_state.user.get('first_name', '')} {st.session_state.user.get('last_name', '')}".strip(),
                "preferences": st.session_state.user.get('preferences', {}),
                "location": user_location
            },
            "nearby_stores": [
                {
                    "name": store["name"],
                    "type": store["type"],
                    "distance_km": store.get("distance_km", 0),
                    "rating": store.get("rating", 0),
                    "address": store.get("address", {})
                }
                for store in nearby_stores
            ],
            "coupons": [
                {
                    "store_name": next(
                        (s["name"] for s in nearby_stores if s["_id"] == coupon["store_id"]), 
                        "a local store"
                    ),
                    "description": coupon["description"],
                    "valid_until": coupon["valid_until"].strftime("%B %d, %Y"),
                    "code": coupon["code"]
                }
                for coupon in coupons
            ]
        }
        
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": """You are a helpful customer support assistant for a chain of coffee shops. 
                Your goal is to provide personalized, helpful responses based on the user's context.
                
                User Context:
                - Name: {user[name]}
                - Location: {user[location][city]}, {user[location][state]}
                - Preferences: {user[preferences]}
                
                Nearby Stores:
                {formatted_stores}
                
                Available Coupons:
                {formatted_coupons}
                
                Guidelines:
                1. Be friendly, professional, and concise.
                2. Use the user's name when appropriate.
                3. Reference nearby stores and available coupons when relevant.
                4. If the user asks about menu items, provide suggestions based on their preferences.
                5. If the user is looking for a store, provide directions and hours.
                6. If the user has a complaint, apologize and offer to help resolve the issue.
                """.format(
                    user=context["user"],
                    formatted_stores="\n".join(
                        f"- {store['name']} ({store['distance_km']:.1f} km away, rating: {store['rating']}/5)"
                        for store in context["nearby_stores"]
                    ),
                    formatted_coupons="\n".join(
                        f"- {coupon['description']} at {coupon['store_name']} (valid until {coupon['valid_until']})"
                        for coupon in context["coupons"]
                    ) if context["coupons"] else "No coupons available"
                )
            },
            {"role": "user", "content": masked_input}
        ]
        
        # Call the LLM
        response = llm_client.call_llm(
            messages,
            temperature=0.7,
            max_tokens=500
        )
        
        # Unmask any PII in the response
        response = unmask_pii(response)
        
        # Store the conversation in the database
        db.chat_history.insert_many([
            {
                "user_id": st.session_state.user["_id"],
                "role": "user",
                "content": user_input,
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "pii_masked": masked_input != user_input,
                    "context": context
                }
            },
            {
                "user_id": st.session_state.user["_id"],
                "role": "assistant",
                "content": response,
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "context_used": {
                        "stores_mentioned": [s["name"] for s in context["nearby_stores"]],
                        "coupons_mentioned": [c["code"] for c in context["coupons"]]
                    }
                }
            }
        ])
        
        # Update chat history in session state
        st.session_state.chat_history = list(db.chat_history.find(
            {"user_id": st.session_state.user["_id"]}
        ).sort("timestamp", -1).limit(20))
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

def render_chat_message(role: str, content: str, timestamp: datetime = None):
    """Render a chat message with appropriate styling."""
    if role == "user":
        avatar = "👤"
        css_class = "user"
    else:
        avatar = "🤖"
        css_class = "assistant"
    
    timestamp_str = f"<small>{timestamp.strftime('%I:%M %p')}</small>" if timestamp else ""
    
    st.markdown(f"""
        <div class="chat-message {css_class}">
            <div class="avatar">{avatar}</div>
            <div class="content">
                {content}
                <div style="text-align: right; margin-top: 0.5rem; font-size: 0.8em; color: #666;">
                    {timestamp}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_login_form():
    """Render the login form."""
    with st.form("login_form"):
        st.subheader("Login to Your Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_clicked = st.form_submit_button("Login")
        with col2:
            signup_clicked = st.form_submit_button("Create Account")
        
        if login_clicked and email and password:
            login_user(email, password)
        elif signup_clicked:
            st.session_state.show_login = False
            st.session_state.show_signup = True
            st.rerun()

def render_signup_form():
    """Render the signup form."""
    with st.form("signup_form"):
        st.subheader("Create an Account")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name")
        with col2:
            last_name = st.text_input("Last Name")
            
        email = st.text_input("Email")
        
        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input("Password", type="password")
        with col2:
            confirm_password = st.text_input("Confirm Password", type="password")
        
        # Preferences
        st.subheader("Preferences")
        favorite_drink = st.selectbox(
            "Favorite Drink",
            ["Coffee", "Tea", "Hot Chocolate", "Iced Coffee", "Other"]
        )
        
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions",
            ["None", "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut Allergy"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Create Account")
        with col2:
            cancel = st.form_submit_button("Back to Login")
        
        if submit and password == confirm_password:
            user_data = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "password": password,
                "confirm_password": confirm_password,
                "preferences": {
                    "favorite_drink": favorite_drink,
                    "dietary_restrictions": dietary_restrictions
                }
            }
            signup_user(user_data)
        elif cancel:
            st.session_state.show_signup = False
            st.session_state.show_login = True
            st.rerun()

def render_chat_interface():
    """Render the main chat interface."""
    # Sidebar with user info and controls
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Coffee+Shop", width=150)
        
        if st.session_state.user:
            st.subheader(f"Welcome, {st.session_state.user.get('first_name', 'User')}!")
            
            # Show usage stats
            usage = rate_limiter.get_usage(st.session_state.user["_id"])
            st.caption(f"API Usage: {usage['current_usage']}/{usage['max_calls']} calls today")
            st.progress(min(1.0, usage['current_usage'] / usage['max_calls']))
            
            if usage['reset_in_seconds']:
                reset_time = (datetime.now() + timedelta(seconds=usage['reset_in_seconds'])).strftime("%I:%M %p")
                st.caption(f"Resets at: {reset_time}")
            
            st.divider()
            
            # Use a form for quick actions to handle state properly
            with st.form("quick_actions"):
                if st.form_submit_button("Find Nearby Stores"):
                    st.session_state.pending_message = "What coffee shops are near me?"
                    st.rerun()
                
                if st.form_submit_button("My Orders"):
                    st.session_state.pending_message = "Show me my recent orders"
                    st.rerun()
                    
                if st.form_submit_button("Available Coupons"):
                    st.session_state.pending_message = "Do I have any coupons?"
                    st.rerun()
            
            st.divider()
            
            if st.button("Logout"):
                logout_user()
    
    # Check for pending messages first
    if st.session_state.pending_message:
        process_user_message(st.session_state.pending_message)
        st.session_state.pending_message = None
        st.rerun()
    
    # Main chat area
    st.title("Customer Support")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            # Display messages in reverse chronological order
            for msg in reversed(st.session_state.chat_history):
                render_chat_message(
                    msg["role"],
                    msg["content"],
                    msg.get("timestamp")
                )
        else:
            st.info("Start a conversation by typing a message below.")
    
    # Chat input - using a form to handle the input state properly
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Type your message...",
                key="user_input",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.form_submit_button("Send", use_container_width=True)
    
    # Process the message if the form is submitted and there's input
    if send_button and user_input.strip():
        # Process the message
        response = process_user_message(user_input.strip())
        # Rerun to update the UI
        st.rerun()

def main():
    """Main application entry point."""
    try:
        # Show the appropriate screen based on authentication state
        if st.session_state.show_signup:
            render_signup_form()
        elif st.session_state.show_login:
            render_login_form()
        elif st.session_state.show_chat and st.session_state.user:
            render_chat_interface()
        else:
            st.session_state.show_login = True
            st.rerun()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Error in main application:")

if __name__ == "__main__":
    main()
