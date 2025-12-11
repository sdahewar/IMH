# 📞 IndiaMART Insights Engine

> Voice Call Analysis Pipeline | Data Voice Hackathon 2024

Transform 5,600+ customer service call transcripts into actionable business insights using **NVIDIA NIM Nemotron-4-Mini-Hindi** - an AI model specifically optimized for Hindi and Hinglish text.

---

## 🎯 Problem Statement

IndiaMART handles millions of sales, service, onboarding, payment, and support calls. Each call contains valuable signals: customer needs, product issues, objections, behavioral trends, common failures, and opportunities. This system extracts actionable insights from these call transcripts.

---

## 📁 Project Structure

```
IMH/
├── main.py                 # Main entry point - run pipeline
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py           # Configuration & categories
│   ├── classifiers/        # Gemini classification
│   │   ├── __init__.py
│   │   └── gemini_classifier.py
│   ├── aggregators/        # Insights aggregation
│   │   ├── __init__.py
│   │   └── insights_aggregator.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── data_loader.py
│       └── helpers.py
│
├── scripts/                # Standalone scripts
│   ├── run_demo.py         # 5 use case demo
│   └── explore_data.py     # Data exploration
│
├── data/                   # Input data (put Excel file here)
├── output/                 # Generated outputs
└── checkpoints/            # Batch processing checkpoints
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. API Key Configuration

The NVIDIA NIM API key is pre-configured in `src/config.py`. 

To use your own key, either:
```powershell
# Option 1: Set environment variable
$env:NVIDIA_API_KEY = "your_api_key_here"

# Option 2: Update src/config.py directly
```

### 3. Run the Pipeline

```powershell
# Quick test (5 samples) - No API key needed
python main.py --mode quick-insights

# Sample classification (10 calls)
python main.py --mode sample --sample-size 10

# Full demo with 5 diverse use cases
python main.py --mode demo

# Process all 5,600+ calls
python main.py --mode full
```

### 4. Launch Agent Pipeline (Interactive)

```powershell
python agent_pipeline.py
```

### 5. Launch GUI Application

```powershell
python gui_app.py
```

### 6. Launch Web Dashboard

```powershell
streamlit run app.py
```

---

## 🤖 Agent Pipeline

The **Agent Pipeline** provides an interactive, LLM-powered interface for transcript analysis.

### Usage Options

```powershell
# Interactive mode (menu-driven)
python agent_pipeline.py

# Analyze by customer type
python agent_pipeline.py --type CATALOG --sample-size 30

# Analyze by city/location
python agent_pipeline.py --city "Makrana" --sample-size 30

# Analyze by customer ID
python agent_pipeline.py --customer 12345

# Analyze a single transcript
python agent_pipeline.py --transcript
```

### Agent Capabilities

| Agent | Function |
|-------|----------|
| **InsightsAgent** | Analyzes individual transcripts, answers questions, extracts action items |
| **AggregationAgent** | Aggregates insights across multiple transcripts by customer/location/type |

### Interactive Mode Features

1. **📝 Single Transcript Analysis** - Paste any transcript and get instant insights
2. **👤 Customer Analysis** - Aggregate all calls for a specific customer
3. **📍 Location Analysis** - Analyze patterns for a specific city
4. **👥 Customer Type Analysis** - Insights for CATALOG, TSCATALOG, STAR, etc.
5. **🔍 Keyword Search** - Find and analyze transcripts containing specific keywords
6. **💬 Follow-up Questions** - Ask the AI questions about analyzed transcripts

---

## 🏷️ Classification Categories (14 Total)

| Category | Description |
|----------|-------------|
| LEAD_QUALITY | Fake/irrelevant leads complaints |
| PAYMENT_BILLING | Refunds, invoices, pricing issues |
| CATALOG_MANAGEMENT | Product listing, images, descriptions |
| SUBSCRIPTION_RENEWAL | Renewals, upgrades, plan queries |
| TECHNICAL_ISSUES | App/website bugs, login problems |
| BUYLEAD_CONSUMPTION | Credits, consumption, deductions |
| ACCOUNT_MANAGEMENT | Profile, GST, verification |
| SERVICE_ESCALATION | Escalated complaints |
| ONBOARDING_TRAINING | New customer guidance |
| CANCELLATION_CHURN | Churn signals, cancellation requests |
| COMPETITOR_MENTION | Competitor comparisons |
| POSITIVE_FEEDBACK | Customer appreciation |
| FOLLOW_UP_REQUIRED | Pending actions needed |
| MISCELLANEOUS | Other issues |

---

## 📊 What Gets Extracted

For each call transcript, the system extracts:

```json
{
  "primary_category": "LEAD_QUALITY",
  "secondary_categories": ["SUBSCRIPTION_RENEWAL"],
  "issue_summary": "Customer complains about irrelevant leads",
  "customer_pain_points": ["Getting leads from wrong city"],
  "resolution_status": "PARTIALLY_RESOLVED",
  "sentiment": "NEGATIVE",
  "sentiment_shift": "IMPROVED",
  "urgency": "HIGH",
  "churn_risk": "MEDIUM",
  "executive_performance": {
    "empathy_shown": true,
    "solution_offered": true,
    "followed_process": true,
    "escalation_needed": false
  },
  "actionable_insight": "Implement city-level lead filtering",
  "requires_follow_up": true,
  "follow_up_reason": "Customer expects callback"
}
```

---

## 💡 Key Insights Generated

### From Quick Analysis (No API Required)

| Metric | Value |
|--------|-------|
| Total Calls | 5,630 |
| Alert Calls | 634 (11.3%) |
| Repeat Ticket Rate | 37% |

### Top Issues Mentioned
1. **BuyLeads** - 1,744 mentions
2. **Subscription** - 664 mentions
3. **Refund** - 497 mentions
4. **Catalog** - 415 mentions
5. **WhatsApp** - 305 mentions

---

## 🎯 Actionable Recommendations

1. **🔥 Address BuyLead Quality** - 30%+ calls mention lead issues
2. **🔄 Reduce 37% Repeat Rate** - Focus on first-call resolution
3. **💰 Automate Refunds** - Self-service for eligible cases
4. **📍 Geographic Focus** - Dedicated support for high-volume cities
5. **⚠️ Churn Prevention** - Proactive outreach for at-risk customers

---

## 🛠️ Tech Stack

- **AI Model**: NVIDIA NIM Nemotron-4-Mini-Hindi-4B-Instruct
  - Specifically optimized for Hindi/Hinglish text
  - OpenAI-compatible API interface
- **Language**: Python 3.10+
- **Dashboard**: Streamlit + Plotly
- **Data Processing**: Pandas
- **API Client**: OpenAI Python SDK
- **Rate Limiting**: Tenacity (auto-retry)

---

## 📈 Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `output/classified_*.csv` | All calls with AI classifications |
| `output/insights_*.json` | Aggregated business insights |
| `output/demo_results_*.json` | Demo use case results |
| `checkpoints/batch_*.json` | Batch processing checkpoints |

---

## 👥 Team

Data Voice Hackathon 2024

---

## 📝 License

MIT License - Feel free to use and modify.

