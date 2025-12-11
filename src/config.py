"""
Configuration settings for the IndiaMART Insights Engine
AI-Assisted Sales & Servicing Enhancement System
"""

# =============================================================================
# INDIAMART BUSINESS CONTEXT
# =============================================================================
"""
PHILOSOPHY: AI helps increase productivity of sales/servicing teams, NOT replace them.
- AI suggests best insights
- Sales/Servicing people know how to take insights to customers
"""

# =============================================================================
# ISSUE CATEGORIES FOR CLASSIFICATION (IndiaMART Specific)
# =============================================================================

ISSUE_CATEGORIES = {
    # Core Product Issues
    "LEAD_QUALITY": {
        "name": "BuyLead Quality Issues",
        "description": "BL quality problems, irrelevant leads, spam leads, past leads not taken up",
        "keywords": ["fake leads", "irrelevant leads", "wrong leads", "BL quality", "spam", "past leads"]
    },
    "PAYMENT_BILLING": {
        "name": "Payment & Billing",
        "description": "Payment delays, invoice issues, refunds, subscription charges",
        "keywords": ["payment", "refund", "invoice", "billing", "price", "charges", "money", "delay"]
    },
    "CATALOG_MANAGEMENT": {
        "name": "Catalog & Listing Issues",
        "description": "Product listing, images, CQS, A Rank, D Rank, MCAT issues, ISQ",
        "keywords": ["catalog", "listing", "images", "CQS", "rank", "MCAT", "ISQ", "product addition"]
    },
    "SUBSCRIPTION_RENEWAL": {
        "name": "Subscription & Renewal",
        "description": "Subscription plans, renewals, upgrades, plan features, upsell opportunities",
        "keywords": ["subscription", "renewal", "upgrade", "plan", "validity", "package"]
    },
    
    # Technical & Platform Issues
    "TECHNICAL_ISSUES": {
        "name": "Technical Problems",
        "description": "App/website bugs, Seller Panel issues, LMS problems, login issues",
        "keywords": ["app", "website", "login", "OTP", "error", "bug", "Seller Panel", "LMS"]
    },
    "BUYLEAD_CONSUMPTION": {
        "name": "BuyLead Consumption",
        "description": "BL credits, consumption, shortlisting, callbacks, replies tracking",
        "keywords": ["buylead", "BL", "credits", "consumption", "shortlist", "callback", "reply"]
    },
    
    # Seller Education & Onboarding
    "SELLER_EDUCATION": {
        "name": "Seller Education Needed",
        "description": "Seller needs training on Seller Panel, LMS, BLs, DIY catalog, product addition",
        "keywords": ["how to", "don't know", "explain", "training", "understand", "learn", "teach"]
    },
    "ONBOARDING_ISSUES": {
        "name": "Onboarding & Verification",
        "description": "OVP issues, verification hurdles, new seller setup, documentation",
        "keywords": ["onboarding", "verification", "OVP", "document", "new seller", "setup"]
    },
    
    # Churn & Retention
    "CHURN_RISK": {
        "name": "Churn Risk Signals",
        "description": "Discontinuation intent, dissatisfaction, competitor mentions, winback needed",
        "keywords": ["cancel", "discontinue", "stop", "not working", "waste", "competitor"]
    },
    "RETENTION_OPPORTUNITY": {
        "name": "Retention Enhancement",
        "description": "Engaged but not renewing, needs intervention, winback opportunity",
        "keywords": ["not renewing", "thinking", "maybe", "will see", "next time"]
    },
    
    # Service Issues
    "SERVICE_ESCALATION": {
        "name": "Service Escalation",
        "description": "Escalated complaints, repeated tickets, unresolved issues, TAT breaches",
        "keywords": ["escalation", "manager", "senior", "complaint", "not resolved", "repeated", "ticket"]
    },
    "PRODUCTION_SUPPORT": {
        "name": "Production Team Support Needed",
        "description": "Catalog enrichment needed, image upload help, professional catalog creation",
        "keywords": ["production", "catalog", "images", "help needed", "professional", "enhance"]
    },
    
    # Opportunities
    "UPSELL_OPPORTUNITY": {
        "name": "Upsell Opportunity",
        "description": "Seller shows interest in higher plans, more features, expansion",
        "keywords": ["upgrade", "more leads", "premium", "better plan", "expand", "grow"]
    },
    "POSITIVE_FEEDBACK": {
        "name": "Positive Feedback",
        "description": "Satisfaction, success stories, good experience, orders received",
        "keywords": ["thank you", "happy", "satisfied", "good", "excellent", "order received", "working well"]
    },
    
    # Follow-up Required
    "FOLLOW_UP_REQUIRED": {
        "name": "Follow-up Required",
        "description": "Callback needed, pending resolution, WC/WM needed",
        "keywords": ["callback", "follow up", "call again", "will check", "get back", "pending"]
    },
    "MISCELLANEOUS": {
        "name": "Miscellaneous",
        "description": "Other issues not fitting above categories",
        "keywords": []
    }
}

# =============================================================================
# SELLER UNDERTONES (Sentiment Types)
# =============================================================================
SELLER_UNDERTONES = [
    "ANGRY",           # Frustrated, shouting, threatening
    "IRRITATED",       # Annoyed, impatient, curt responses
    "DISSATISFIED",    # Unhappy with service/product
    "DISENGAGED",      # Not interested, giving short answers
    "CONFUSED",        # Doesn't understand product/process
    "HESITANT",        # Uncertain, needs convincing
    "NEUTRAL",         # Normal conversation
    "INTERESTED",      # Showing curiosity, asking questions
    "SATISFIED",       # Happy with service
    "ENTHUSIASTIC"     # Very positive, advocating
]

# =============================================================================
# CATALOG QUALITY METRICS
# =============================================================================
CATALOG_METRICS = {
    "CQS": "Catalog Quality Score",
    "A_RANK": "High engagement MCAT (>=3 transactions in 6 months, 1 in 2 weeks)",
    "D_RANK": "Low engagement MCAT, needs primary products",
    "ISQ": "Image/Specification Quality",
    "POP": "Product on Platform",
    "MCAT": "Micro-category"
}

# =============================================================================
# TRANSACTION DEFINITIONS (For A Rank)
# =============================================================================
TRANSACTION_TYPES = [
    "BL Consumption",
    "Callbacks",
    "Replies (3 Replies)",
    "I am Interested (on Live BL)",
    "Bizfeed Transactions",
    "Bizfeed NIs",
    "I'm Interested clicks on sold-out BLs"
]

# =============================================================================
# REACH OUT METHODS
# =============================================================================
REACH_OUT_METHODS = [
    "Email",
    "WhatsApp",
    "Seller Panel",
    "IM App",
    "DIR (Directory)",
    "Help Page",
    "Phone Call"
]

# =============================================================================
# NVIDIA NIM MODEL CONFIGURATION
# =============================================================================
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-4-mini-hindi-4b-instruct"
NVIDIA_API_KEY = "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"

# Model parameters
MODEL_TEMPERATURE = 0.3
MODEL_TOP_P = 0.9
MODEL_MAX_TOKENS = 4096  # Increased for detailed insights

# Batch processing
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
DATA_DIR = "data"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = "checkpoints"

# =============================================================================
# OUTPUT FILES
# =============================================================================
CLASSIFIED_OUTPUT = "classified_calls.csv"
INSIGHTS_OUTPUT = "insights_report.json"
DASHBOARD_OUTPUT = "dashboard_data.json"
