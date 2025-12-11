"""
Configuration settings for the IndiaMART Insights Engine
AI-Assisted Sales & Servicing Enhancement System
"""

# =============================================================================
# NVIDIA NIM MODEL CONFIGURATION
# =============================================================================
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-4-mini-hindi-4b-instruct"
NVIDIA_API_KEY = "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"

# Model parameters
MODEL_TEMPERATURE = 0.3
MODEL_TOP_P = 0.9
MODEL_MAX_TOKENS = 4096

# Batch processing
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 2

# =============================================================================
# ISSUE CATEGORIES FOR CLASSIFICATION (IndiaMART Specific)
# =============================================================================

ISSUE_CATEGORIES = {
    "LEAD_QUALITY": "BuyLead Quality Issues - BL quality problems, irrelevant leads, spam leads",
    "PAYMENT_BILLING": "Payment & Billing - Payment delays, invoice issues, refunds",
    "CATALOG_MANAGEMENT": "Catalog & Listing Issues - Product listing, CQS, A Rank, D Rank, MCAT",
    "SUBSCRIPTION_RENEWAL": "Subscription & Renewal - Plans, renewals, upgrades",
    "TECHNICAL_ISSUES": "Technical Problems - App/website bugs, Seller Panel, LMS issues",
    "BUYLEAD_CONSUMPTION": "BuyLead Consumption - BL credits, consumption, shortlisting",
    "SELLER_EDUCATION": "Seller Education Needed - Training on platform features",
    "ONBOARDING_ISSUES": "Onboarding & Verification - OVP issues, new seller setup",
    "CHURN_RISK": "Churn Risk Signals - Discontinuation intent, competitor mentions",
    "RETENTION_OPPORTUNITY": "Retention Enhancement - Engaged but not renewing",
    "SERVICE_ESCALATION": "Service Escalation - Escalated complaints, unresolved issues",
    "PRODUCTION_SUPPORT": "Production Team Support - Catalog enrichment needed",
    "UPSELL_OPPORTUNITY": "Upsell Opportunity - Interest in higher plans",
    "POSITIVE_FEEDBACK": "Positive Feedback - Satisfaction, success stories",
    "FOLLOW_UP_REQUIRED": "Follow-up Required - Callback needed, pending resolution",
    "MISCELLANEOUS": "Miscellaneous - Other issues"
}

# =============================================================================
# SELLER UNDERTONES
# =============================================================================
SELLER_UNDERTONES = [
    "ANGRY", "IRRITATED", "DISSATISFIED", "DISENGAGED", "CONFUSED",
    "HESITANT", "NEUTRAL", "INTERESTED", "SATISFIED", "ENTHUSIASTIC"
]

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
DATA_DIR = "data"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = "checkpoints"
CLASSIFIED_OUTPUT = "classified_calls.csv"
INSIGHTS_OUTPUT = "insights_report.json"
