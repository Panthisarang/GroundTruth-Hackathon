# 🚀 H-002 Customer Experience Automation

**Tagline:** A hyper-personalized retail assistant that transforms vague customer messages into location-aware, preference-aware, context-rich actions — powered by RAG, real-time data, and privacy-safe AI.

---

# 1. The Problem (Real World Scenario)

### **Context**

Modern retail brands receive thousands of customer messages every day:

* “I’m cold.”
* “What can I get quickly?”
* “Where’s the nearest store with coffee?”
* “Do I have any coupons left?”
* “Where is my order?”

Standard chatbots respond with:

> *“Can you clarify your issue?”*

But customers don't want clarification — they want **solutions**, immediately.

### **The Pain Point**

Retail brands suffer from:

* **Inefficient support agents** asking repetitive questions (“What’s your email?”, “Where are you?”)
* **Lost sales** because chatbots fail to convert vague intent into purchases
* **Poor experience** due to irrelevant recommendations
* **Slow response loops** (especially for location-based or offer-based queries)

### **My Solution**

I built **H-002 Customer Experience Automation**, a full-stack, production-minded system that:

💬 Interprets vague user messages
📍 Combines them with real-time location
🧾 Reads internal policy & FAQs using RAG
🛍️ Checks user purchase history
🎟️ Fetches personalized coupons
🔒 Masks all PII before LLM usage
⚡ Returns a precise, contextual, actionable response

Example:
**User:** *“I’m cold.”*
**AI:**

> “You’re 43m from Downtown Café. They have Hot Chocolate at 15% off. Want me to apply the coupon?”

This is not a chatbot —
It’s an **intelligent micro-journey engine** tailored for retail.

---

# 2. Expected End Result (User Experience)

### **Input**

User logs into the Streamlit app, opens chat, and types a vague natural-language message like:

* “I’m hungry”
* “Where should I go?”
* “Any offers?”

### **Action**

The system automatically:

1. Fetches the user profile from MongoDB
2. Loads order history + preferences
3. Retrieves nearby stores based on a simulated location
4. Pulls available coupons
5. Runs RAG retrieval from policy & FAQ documents
6. Masks all PII
7. Sends a compact structured prompt to the LLM
8. Generates a short, actionable answer

### **Output**

A rich, actionable message that may include:

* Nearest store and walking distance
* Personalized coupon
* Recommended items based on preferences
* Relevant policy snippet (refund, privacy, order tracking)
* Upsell opportunities (“Would you like a warm drink?”)

**All inside a modern Streamlit UI with history, personalization, and API usage tracking.**

---

# 3. Technical Approach (Production-Ready Design)

I wanted this project to go beyond a simple chatbot.
I built a **modular, realistic, production-aligned system** with privacy, rate-limiting, retrieval, and orchestration.

---

## ✔ **Pipeline Overview**

### **1. Data Layer (MongoDB)**

The database stores:

* Users
* Orders
* Stores
* Coupons
* Embeddings
* Chat history
* Daily LLM usage counters

### **2. PII Masking Layer**

Before sending **any text** to LLM:

* Emails → `<EMAIL_1>`
* Phone numbers → `<PHONE_1>`
* Card numbers → `<CARD_1>`
* Addresses → `<ADDRESS_1>`

Supports **deterministic masking** so user experience remains consistent.

### **3. Retrieval-Augmented Generation (RAG)**

Internal policies & FAQs are chunked → embedded → stored → retrieved in real time.

Prevents hallucinations and ensures compliance.

### **4. Context Enrichment**

We build a structured prompt using:

* Masked user query
* Nearest store (Haversine distance)
* Coupons
* Preferences
* Order history summary
* RAG snippets

### **5. LLM Layer**

A lightweight model (gpt-3.5 / gpt-4.1-mini) generates:

* Short
* Actionable
* Personalized
* Safe responses

A smart retry + limit-aware client ensures:

* No socket explosion
* No infinite retries
* No excessive token usage

### **6. UI Layer (Streamlit)**

Shows:

* Chat bubbles
* Sidebar with profile
* API usage bar
* Debug toggle for retrieved context

---

## ✔ **System Architecture**

```
User → Streamlit UI → Auth → Chat Engine → PII Masker → RAG → Context Builder → LLM Client → Response
```

Modules communicate cleanly using:

* `db.py` for all database IO
* `llm_client.py` for all model calls
* `rag_pipeline.py` for document retrieval
* `rate_limiter.py` for quota governance
* `utils.py` for summarization + geolocation

A pipeline built to be **extendable, auditable, and reliable**.

---

# 4. Tech Stack

### **Language**

* Python 3.10

### **Backend**

* Streamlit
* Python (modular architecture)
* Pydantic

### **Database**

* MongoDB (PyMongo)

### **Retrieval**

* Sentence Transformers (embeddings)
* Simple vector similarity (MongoDB stored norms)

### **AI Model**

* OpenAI GPT-3.5 / GPT-4.1-mini
* (Swappable: Groq, Gemini, OpenRouter)

### **Security**

* BCrypt hashed passwords
* PII masking before LLM
* Daily per-user rate limits

### **UX**

* Custom CSS
* Chat UI
* Sidebar with live usage stats

### **Utilities**

* Faker
* Pypdf
* Haversine formula

---

# 5. Challenges & Learnings

### **Challenge 1 — PII Exposure Risk**

Problem:
LLMs should *never* see raw user phone numbers, emails, or order IDs.

Solution:
A deterministic PII masking module + system prompt guard that forbids reconstruction.

---

### **Challenge 2 — API Rate Limits & Network Stability**

Problem:
Windows aggressively throttles socket creation → hitting OpenAI limits caused:

* `ConnectionResetError 10054`
* `WinError 10055`

Solution:

* Added **bounded retry logic**
* Prevented recursive retry loops
* Added local mocking mode (`MOCK_LLM`)
* Added per-user daily quota

---

### **Challenge 3 — RAG Relevance**

Initially, RAG fetched noisy chunks.

Solution:

* Tighter chunk sizes
* Improved embedding metadata
* Retrieved only top-3 snippets for token efficiency

---

### **Challenge 4 — Generating Personalized Responses**

Not just “What do you want?”
But:

* Considering user history
* Considering location
* Considering store offers

Solution:
A structured system prompt that guides the LLM to behave like:

**“A hyper-personalized retail concierge.”**

---

# 6. Visual Proof (What Reviewers Will See)

### ✔ Login & Signup Page

Beautiful Streamlit UI with validation and secure password hashing.

### ✔ Chat Interface

Human-like bubbles with:

* User messages (right aligned)
* AI responses (left aligned)

### ✔ Sidebar

Displays:

* User profile
* Preferences
* API usage bar
* Retrieved RAG snippets

### ✔ Database Snapshots

MongoDB collection for:

* users
* coupons
* stores
* orders
* chat_history

---

# 7. How to Run

```bash
# 1. Clone Repository
git clone https://github.com/username/H002-Customer-Experience-Automation.git
cd H002-Customer-Experience-Automation

# 2. Create venv
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add .env
# (Mongo + OpenAI keys)
notepad .env

# 5. Generate synthetic dataset
python -m h002_customer_experience_automation.data_generator

# 6. Run the app
streamlit run h002_customer_experience_automation/app.py
```

---

# 8. Project Structure

```
h002_customer_experience_automation/
│── app.py
│── auth.py
│── config.py
│── db.py
│── llm_client.py
│── pii_masker.py
│── rag_pipeline.py
│── rate_limiter.py
│── data_generator.py
│── utils.py
└── README.md
```

