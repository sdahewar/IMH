"""
NVIDIA NIM Classifier for Hinglish Call Transcripts
Uses Nemotron-4-Mini-Hindi model optimized for Hindi/Hinglish text
Handles batch processing of 5600+ transcripts with rate limiting and error handling
"""

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    ISSUE_CATEGORIES, 
    NVIDIA_BASE_URL, 
    NVIDIA_MODEL, 
    NVIDIA_API_KEY,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_MAX_TOKENS,
    BATCH_SIZE, 
    MAX_RETRIES
)

# =============================================================================
# CLASSIFICATION PROMPT TEMPLATE
# =============================================================================

CLASSIFICATION_PROMPT = """You are an expert customer service analyst for IndiaMART, India's largest B2B marketplace. 
Analyze the following call transcript (in Hinglish - Hindi-English mix) and extract structured insights.

TRANSCRIPT:
{transcript}

CALL METADATA:
- Customer Type: {customer_type}
- City: {city}
- Call Direction: {call_direction}
- Is Repeat Ticket: {is_repeat}
- Call Duration: {duration} seconds

EXISTING SUMMARY (if available):
{summary}

Analyze this transcript and provide a JSON response with the following structure:

{{
    "primary_category": "<One of: LEAD_QUALITY, PAYMENT_BILLING, CATALOG_MANAGEMENT, SUBSCRIPTION_RENEWAL, TECHNICAL_ISSUES, BUYLEAD_CONSUMPTION, ACCOUNT_MANAGEMENT, SERVICE_ESCALATION, ONBOARDING_TRAINING, CANCELLATION_CHURN, COMPETITOR_MENTION, POSITIVE_FEEDBACK, FOLLOW_UP_REQUIRED, MISCELLANEOUS>",
    "secondary_categories": ["<list of other applicable categories>"],
    "issue_summary": "<Brief 1-2 sentence summary of the main issue in English>",
    "customer_pain_points": ["<list of specific pain points expressed>"],
    "resolution_status": "<RESOLVED | PARTIALLY_RESOLVED | UNRESOLVED | ESCALATED | CALLBACK_SCHEDULED>",
    "sentiment": "<VERY_NEGATIVE | NEGATIVE | NEUTRAL | POSITIVE | VERY_POSITIVE>",
    "sentiment_shift": "<IMPROVED | DECLINED | STABLE>",
    "urgency": "<CRITICAL | HIGH | MEDIUM | LOW>",
    "churn_risk": "<HIGH | MEDIUM | LOW | NONE>",
    "executive_performance": {{
        "empathy_shown": <true/false>,
        "solution_offered": <true/false>,
        "followed_process": <true/false>,
        "escalation_needed": <true/false>
    }},
    "actionable_insight": "<One specific actionable recommendation based on this call>",
    "keywords": ["<relevant keywords for this call>"],
    "requires_follow_up": <true/false>,
    "follow_up_reason": "<if requires follow up, why?>"
}}

IMPORTANT GUIDELINES:
1. Focus on actionable insights that can improve customer experience
2. Identify patterns that indicate systemic issues
3. Be precise with category classification
4. Consider the Hinglish context - many issues are expressed in mixed language
5. Pay attention to customer frustration signals even if not explicitly stated
6. Look for competitor mentions or churn signals
7. Assess if the executive handled the call professionally

Respond ONLY with valid JSON, no additional text."""


# =============================================================================
# NVIDIA NIM CLIENT CLASS
# =============================================================================

class NvidiaClassifier:
    """
    Handles classification of call transcripts using NVIDIA NIM Nemotron-4-Mini-Hindi
    Optimized for Hinglish (Hindi-English) transcripts
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the NVIDIA NIM client"""
        self.api_key = api_key or NVIDIA_API_KEY
        self.base_url = base_url or NVIDIA_BASE_URL
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.model = NVIDIA_MODEL
        
    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
    def classify_single(self, transcript: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single transcript
        
        Args:
            transcript: The call transcript text
            metadata: Dictionary containing call metadata
            
        Returns:
            Dictionary with classification results
        """
        prompt = CLASSIFICATION_PROMPT.format(
            transcript=transcript[:6000],  # Limit transcript length for model context
            customer_type=metadata.get('customer_type', 'Unknown'),
            city=metadata.get('city', 'Unknown'),
            call_direction=metadata.get('call_direction', 'Unknown'),
            is_repeat=metadata.get('is_repeat', 'Unknown'),
            duration=metadata.get('duration', 'Unknown'),
            summary=metadata.get('summary', 'Not available')[:800]
        )
        
        try:
            # Use streaming for better handling
            response_text = ""
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=MODEL_TEMPERATURE,
                top_p=MODEL_TOP_P,
                max_tokens=MODEL_MAX_TOKENS,
                stream=True
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            
            # Clean response - handle potential markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Extract content between code blocks
                lines = response_text.split("\n")
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if i > 0 and line.strip() == "```":
                        end_idx = i
                        break
                response_text = "\n".join(lines[start_idx:end_idx])
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            # Parse JSON
            result = json.loads(response_text)
            result['classification_success'] = True
            return result
            
        except json.JSONDecodeError as e:
            return {
                'classification_success': False,
                'error': f'JSON parse error: {str(e)}',
                'primary_category': 'MISCELLANEOUS',
                'raw_response': response_text[:500] if 'response_text' in locals() else 'No response'
            }
        except Exception as e:
            return {
                'classification_success': False,
                'error': str(e),
                'primary_category': 'MISCELLANEOUS'
            }
    
    def classify_single_non_streaming(self, transcript: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single transcript (non-streaming version)
        
        Args:
            transcript: The call transcript text
            metadata: Dictionary containing call metadata
            
        Returns:
            Dictionary with classification results
        """
        prompt = CLASSIFICATION_PROMPT.format(
            transcript=transcript[:6000],
            customer_type=metadata.get('customer_type', 'Unknown'),
            city=metadata.get('city', 'Unknown'),
            call_direction=metadata.get('call_direction', 'Unknown'),
            is_repeat=metadata.get('is_repeat', 'Unknown'),
            duration=metadata.get('duration', 'Unknown'),
            summary=metadata.get('summary', 'Not available')[:800]
        )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=MODEL_TEMPERATURE,
                top_p=MODEL_TOP_P,
                max_tokens=MODEL_MAX_TOKENS,
                stream=False
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Clean response
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                start_idx = 1
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if i > 0 and line.strip() == "```":
                        end_idx = i
                        break
                response_text = "\n".join(lines[start_idx:end_idx])
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            result = json.loads(response_text)
            result['classification_success'] = True
            return result
            
        except json.JSONDecodeError as e:
            return {
                'classification_success': False,
                'error': f'JSON parse error: {str(e)}',
                'primary_category': 'MISCELLANEOUS',
                'raw_response': response_text[:500] if 'response_text' in locals() else 'No response'
            }
        except Exception as e:
            return {
                'classification_success': False,
                'error': str(e),
                'primary_category': 'MISCELLANEOUS'
            }
    
    def classify_batch(self, df: pd.DataFrame, 
                       transcript_col: str = 'transcript',
                       progress_callback=None) -> pd.DataFrame:
        """
        Classify all transcripts in a DataFrame
        
        Args:
            df: DataFrame with transcripts
            transcript_col: Name of transcript column
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with added classification columns
        """
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying transcripts"):
            metadata = {
                'customer_type': row.get('customer_type', ''),
                'city': row.get('city_name', ''),
                'call_direction': row.get('FLAG_IN_OUT', ''),
                'is_repeat': row.get('is_ticket_repeat60d', ''),
                'duration': row.get('call_duration', ''),
                'summary': row.get('summary', '')
            }
            
            result = self.classify_single(row[transcript_col], metadata)
            result['original_index'] = idx
            results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
            
            if progress_callback:
                progress_callback(idx + 1, len(df))
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original DataFrame
        df_with_results = df.copy()
        for col in results_df.columns:
            if col != 'original_index':
                df_with_results[f'ai_{col}'] = results_df[col].values
        
        return df_with_results


# =============================================================================
# BATCH PROCESSING WITH CHECKPOINTING
# =============================================================================

class BatchProcessor:
    """
    Handles batch processing with checkpointing for large datasets
    """
    
    def __init__(self, classifier: NvidiaClassifier, checkpoint_dir: str = "checkpoints"):
        self.classifier = classifier
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_path(self, batch_id: int) -> str:
        return os.path.join(self.checkpoint_dir, f"batch_{batch_id}.json")
    
    def load_checkpoint(self, batch_id: int) -> Optional[List[Dict]]:
        path = self.get_checkpoint_path(batch_id)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_checkpoint(self, batch_id: int, results: List[Dict]):
        path = self.get_checkpoint_path(batch_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def process_with_checkpoints(self, df: pd.DataFrame, 
                                  batch_size: int = BATCH_SIZE,
                                  resume: bool = True) -> pd.DataFrame:
        """
        Process DataFrame in batches with checkpointing
        Shows detailed status for each record
        
        Args:
            df: Input DataFrame
            batch_size: Number of records per batch
            resume: Whether to resume from checkpoints
            
        Returns:
            DataFrame with all classifications
        """
        all_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        total_records = len(df)
        
        # Statistics tracking
        stats = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'api_errors': 0,
            'from_checkpoint': 0
        }
        
        print("\n" + "=" * 80)
        print(f"🚀 STARTING FULL CLASSIFICATION")
        print("=" * 80)
        print(f"📊 Total Records: {total_records:,}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"📁 Total Batches: {total_batches}")
        print(f"🤖 Model: {NVIDIA_MODEL}")
        print(f"💾 Checkpoint Directory: {self.checkpoint_dir}")
        print("=" * 80 + "\n")
        
        start_time = time.time()
        
        for batch_id in range(total_batches):
            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_start_time = time.time()
            
            # Check for existing checkpoint
            if resume:
                checkpoint = self.load_checkpoint(batch_id)
                if checkpoint:
                    stats['from_checkpoint'] += len(checkpoint)
                    stats['processed'] += len(checkpoint)
                    for r in checkpoint:
                        if r.get('classification_success'):
                            stats['success'] += 1
                        else:
                            stats['failed'] += 1
                    all_results.extend(checkpoint)
                    print(f"📂 Batch {batch_id + 1}/{total_batches}: Loaded {len(checkpoint)} records from checkpoint")
                    continue
            
            # Process batch
            print(f"\n{'─' * 80}")
            print(f"📦 BATCH {batch_id + 1}/{total_batches} | Records {start_idx + 1} to {end_idx}")
            print(f"{'─' * 80}")
            
            batch_df = df.iloc[start_idx:end_idx]
            batch_results = []
            
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                record_num = start_idx + i + 1
                call_id = row.get('click_to_call_id', 'N/A')
                city = row.get('city_name', 'N/A')
                ctype = row.get('customer_type', 'N/A')
                duration = row.get('call_duration', 0)
                
                # Show record being processed
                print(f"   [{record_num:5}/{total_records}] ID: {call_id} | {city[:15]:15} | {ctype[:10]:10} | {duration:4}s ", end="", flush=True)
                
                metadata = {
                    'customer_type': ctype,
                    'city': city,
                    'call_direction': row.get('FLAG_IN_OUT', ''),
                    'is_repeat': row.get('is_ticket_repeat60d', ''),
                    'duration': duration,
                    'summary': row.get('summary', '')
                }
                
                try:
                    result = self.classifier.classify_single(row['transcript'], metadata)
                    result['original_index'] = idx
                    result['call_id'] = call_id
                    batch_results.append(result)
                    
                    stats['processed'] += 1
                    
                    if result.get('classification_success'):
                        stats['success'] += 1
                        category = result.get('primary_category', 'N/A')
                        sentiment = result.get('sentiment', 'N/A')
                        churn = result.get('churn_risk', 'N/A')
                        print(f"✅ {category[:20]:20} | {sentiment[:10]:10} | Churn: {churn}")
                    else:
                        stats['failed'] += 1
                        error = result.get('error', 'Unknown error')[:40]
                        print(f"⚠️  FAILED: {error}")
                        
                        # Check for API limit errors
                        if 'rate' in error.lower() or 'limit' in error.lower() or '429' in error:
                            stats['api_errors'] += 1
                            print(f"   🚨 API RATE LIMIT DETECTED! Waiting 60 seconds...")
                            time.sleep(60)
                
                except Exception as e:
                    stats['processed'] += 1
                    stats['failed'] += 1
                    error_msg = str(e)[:50]
                    print(f"❌ ERROR: {error_msg}")
                    
                    # Check for API errors
                    if 'rate' in error_msg.lower() or 'limit' in error_msg.lower() or '429' in str(e):
                        stats['api_errors'] += 1
                        print(f"   🚨 API RATE LIMIT! Pausing for 60 seconds...")
                        time.sleep(60)
                    
                    batch_results.append({
                        'original_index': idx,
                        'call_id': call_id,
                        'classification_success': False,
                        'error': str(e),
                        'primary_category': 'MISCELLANEOUS'
                    })
                
                time.sleep(0.3)  # Rate limiting between records
            
            # Save checkpoint
            self.save_checkpoint(batch_id, batch_results)
            all_results.extend(batch_results)
            
            # Batch summary
            batch_time = time.time() - batch_start_time
            elapsed = time.time() - start_time
            remaining_batches = total_batches - batch_id - 1
            eta = (elapsed / (batch_id + 1)) * remaining_batches if batch_id > 0 else 0
            
            print(f"\n   📊 Batch Complete: {len(batch_results)} records in {batch_time:.1f}s")
            print(f"   💾 Checkpoint saved")
            print(f"   ⏱️  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m remaining")
            print(f"   📈 Progress: {stats['processed']:,}/{total_records:,} ({stats['processed']/total_records*100:.1f}%)")
            print(f"   ✅ Success: {stats['success']:,} | ⚠️ Failed: {stats['failed']:,} | 🚨 API Errors: {stats['api_errors']}")
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("🎉 CLASSIFICATION COMPLETE")
        print("=" * 80)
        print(f"📊 Total Processed: {stats['processed']:,}")
        print(f"✅ Successful: {stats['success']:,} ({stats['success']/stats['processed']*100:.1f}%)")
        print(f"⚠️  Failed: {stats['failed']:,}")
        print(f"📂 From Checkpoints: {stats['from_checkpoint']:,}")
        print(f"🚨 API Errors: {stats['api_errors']}")
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"⚡ Avg Speed: {stats['processed']/total_time:.2f} records/second")
        print("=" * 80 + "\n")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        return results_df


# =============================================================================
# QUICK SAMPLE CLASSIFIER (for testing)
# =============================================================================

def classify_sample(df: pd.DataFrame, sample_size: int = 5, api_key: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Quick test classification on a sample of transcripts
    Returns a merged DataFrame with all original columns + LLM metrics
    Shows detailed status for each record
    
    Args:
        df: Full DataFrame
        sample_size: Number of samples to classify
        api_key: Optional API key (uses default if not provided)
        verbose: Show detailed per-record status
        
    Returns:
        DataFrame with all original columns + classification results
    """
    classifier = NvidiaClassifier(api_key=api_key)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    sample_df = sample_df.reset_index(drop=True)
    
    total = len(sample_df)
    stats = {'success': 0, 'failed': 0, 'api_errors': 0}
    
    if verbose:
        print(f"\n{'─' * 80}")
        print(f"📊 Processing {total} records")
        print(f"{'─' * 80}")
    
    results = []
    start_time = time.time()
    
    for idx, row in sample_df.iterrows():
        call_id = row.get('click_to_call_id', 'N/A')
        city = str(row.get('city_name', 'N/A'))[:15]
        ctype = str(row.get('customer_type', 'N/A'))[:10]
        duration = row.get('call_duration', 0)
        
        if verbose:
            print(f"   [{idx + 1:4}/{total}] ID: {call_id} | {city:15} | {ctype:10} | {duration:4}s ", end="", flush=True)
        
        metadata = {
            'customer_type': row.get('customer_type', ''),
            'city': row.get('city_name', ''),
            'call_direction': row.get('FLAG_IN_OUT', ''),
            'is_repeat': row.get('is_ticket_repeat60d', ''),
            'duration': duration,
            'summary': row.get('summary', '')
        }
        
        try:
            result = classifier.classify_single(row['transcript'], metadata)
            results.append(result)
            
            if result.get('classification_success'):
                stats['success'] += 1
                if verbose:
                    category = result.get('primary_category', 'N/A')[:20]
                    sentiment = result.get('sentiment', 'N/A')[:10]
                    churn = result.get('churn_risk', 'N/A')
                    print(f"✅ {category:20} | {sentiment:10} | Churn: {churn}")
            else:
                stats['failed'] += 1
                error = result.get('error', 'Unknown')[:40]
                if verbose:
                    print(f"⚠️  FAILED: {error}")
                
                # Check for API rate limit
                if 'rate' in error.lower() or 'limit' in error.lower() or '429' in error:
                    stats['api_errors'] += 1
                    if verbose:
                        print(f"   🚨 API RATE LIMIT! Waiting 60 seconds...")
                    time.sleep(60)
        
        except Exception as e:
            stats['failed'] += 1
            error_msg = str(e)[:50]
            if verbose:
                print(f"❌ ERROR: {error_msg}")
            
            if 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                stats['api_errors'] += 1
                if verbose:
                    print(f"   🚨 API RATE LIMIT! Waiting 60 seconds...")
                time.sleep(60)
            
            results.append({
                'classification_success': False,
                'error': str(e),
                'primary_category': 'MISCELLANEOUS'
            })
        
        time.sleep(0.3)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"{'─' * 80}")
        print(f"📊 Complete: {stats['success']} success | {stats['failed']} failed | {stats['api_errors']} API errors")
        print(f"⏱️  Time: {elapsed:.1f}s | Speed: {total/elapsed:.2f} records/sec")
        print(f"{'─' * 80}\n")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Flatten nested dictionaries (like executive_performance)
    results_df = flatten_classification_results(results_df)
    
    # Add 'ai_' prefix to all classification columns
    results_df = results_df.add_prefix('ai_')
    
    # Merge with original sample data
    merged_df = pd.concat([sample_df.reset_index(drop=True), results_df], axis=1)
    
    return merged_df


def flatten_classification_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested dictionaries and lists in classification results
    """
    result_df = df.copy()
    
    # Flatten executive_performance dict
    if 'executive_performance' in result_df.columns:
        exec_perf = result_df['executive_performance'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        result_df['exec_empathy_shown'] = exec_perf.apply(lambda x: x.get('empathy_shown', None))
        result_df['exec_solution_offered'] = exec_perf.apply(lambda x: x.get('solution_offered', None))
        result_df['exec_followed_process'] = exec_perf.apply(lambda x: x.get('followed_process', None))
        result_df['exec_escalation_needed'] = exec_perf.apply(lambda x: x.get('escalation_needed', None))
        result_df = result_df.drop(columns=['executive_performance'])
    
    # Convert lists to comma-separated strings
    for col in ['secondary_categories', 'customer_pain_points', 'keywords']:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x else ''
            )
    
    return result_df


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for backward compatibility with existing code
GeminiClassifier = NvidiaClassifier
