"""
IndiaMART Insights Engine - GUI Application
Simple Python GUI using Tkinter for transcript analysis

Run with: python gui_app.py
"""

import os
import sys
import json
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import InsightsAgent, AggregationAgent
from src.config import NVIDIA_MODEL, OUTPUT_DIR


class InsightsEngineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IndiaMART Insights Engine")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        # Data
        self.df = None
        self.insights_agent = None
        self.aggregation_agent = None
        self.current_result = None
        
        # Style configuration
        self.setup_styles()
        
        # Build UI
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Load data on startup
        self.root.after(100, self.load_data)
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'text': '#eaeaea',
            'text_dim': '#a0a0a0',
            'success': '#00d26a',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
        
        style.configure('Header.TLabel', 
                       background=self.colors['bg_dark'], 
                       foreground=self.colors['accent'],
                       font=('Segoe UI', 24, 'bold'))
        
        style.configure('SubHeader.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_dim'],
                       font=('Segoe UI', 10))
        
        style.configure('TButton',
                       font=('Segoe UI', 10),
                       padding=10)
        
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('TNotebook',
                       background=self.colors['bg_dark'])
        
        style.configure('TNotebook.Tab',
                       font=('Segoe UI', 10),
                       padding=[15, 5])
        
        style.configure('TFrame',
                       background=self.colors['bg_dark'])
        
        style.configure('TLabelframe',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text'])
        
        style.configure('TLabelframe.Label',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['accent'],
                       font=('Segoe UI', 11, 'bold'))
    
    def create_header(self):
        """Create header section"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=20, pady=10)
        
        title = ttk.Label(header_frame, 
                         text="🤖 IndiaMART Insights Engine",
                         style='Header.TLabel')
        title.pack(side='left')
        
        subtitle = ttk.Label(header_frame,
                            text=f"Powered by NVIDIA NIM ({NVIDIA_MODEL})",
                            style='SubHeader.TLabel')
        subtitle.pack(side='left', padx=20, pady=5)
        
        # Data info
        self.data_label = ttk.Label(header_frame,
                                   text="📊 Loading data...",
                                   style='SubHeader.TLabel')
        self.data_label.pack(side='right')
    
    def create_main_content(self):
        """Create main content area with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Tab 1: Single Transcript Analysis
        self.create_single_transcript_tab()
        
        # Tab 2: Batch Analysis
        self.create_batch_analysis_tab()
        
        # Tab 3: Results View
        self.create_results_tab()
    
    def create_single_transcript_tab(self):
        """Create single transcript analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📝 Single Transcript")
        
        # Left panel - Input
        left_frame = ttk.LabelFrame(tab, text="Input Transcript", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Transcript input
        self.transcript_input = scrolledtext.ScrolledText(
            left_frame, 
            width=50, 
            height=20,
            font=('Consolas', 10),
            bg='#16213e',
            fg='#eaeaea',
            insertbackground='white'
        )
        self.transcript_input.pack(fill='both', expand=True)
        self.transcript_input.insert('1.0', 'Paste your transcript here...')
        self.transcript_input.bind('<FocusIn>', self.clear_placeholder)
        
        # Metadata frame
        meta_frame = ttk.Frame(left_frame)
        meta_frame.pack(fill='x', pady=10)
        
        ttk.Label(meta_frame, text="Customer Type:").pack(side='left')
        self.cust_type_var = tk.StringVar(value="CATALOG")
        cust_type_combo = ttk.Combobox(meta_frame, textvariable=self.cust_type_var, width=15)
        cust_type_combo['values'] = ('CATALOG', 'TSCATALOG', 'STAR', 'LEADER', 'FREELIST')
        cust_type_combo.pack(side='left', padx=5)
        
        ttk.Label(meta_frame, text="City:").pack(side='left', padx=(20, 0))
        self.city_var = tk.StringVar(value="")
        city_entry = ttk.Entry(meta_frame, textvariable=self.city_var, width=20)
        city_entry.pack(side='left', padx=5)
        
        # Analyze button
        analyze_btn = ttk.Button(left_frame, text="🔍 Analyze Transcript", 
                                command=self.analyze_single_transcript)
        analyze_btn.pack(pady=10)
        
        # Right panel - Results
        right_frame = ttk.LabelFrame(tab, text="Analysis Results", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.single_result_text = scrolledtext.ScrolledText(
            right_frame,
            width=50,
            height=25,
            font=('Consolas', 10),
            bg='#16213e',
            fg='#eaeaea'
        )
        self.single_result_text.pack(fill='both', expand=True)
        
        # Question frame
        q_frame = ttk.Frame(right_frame)
        q_frame.pack(fill='x', pady=5)
        
        ttk.Label(q_frame, text="Ask a question:").pack(side='left')
        self.question_var = tk.StringVar()
        q_entry = ttk.Entry(q_frame, textvariable=self.question_var, width=40)
        q_entry.pack(side='left', padx=5, fill='x', expand=True)
        q_entry.bind('<Return>', lambda e: self.ask_question())
        
        ask_btn = ttk.Button(q_frame, text="Ask", command=self.ask_question)
        ask_btn.pack(side='left', padx=5)
    
    def create_batch_analysis_tab(self):
        """Create batch analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Batch Analysis")
        
        # Create left and right panes
        paned = ttk.PanedWindow(tab, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left pane - Input options
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Input source selection
        source_frame = ttk.LabelFrame(left_frame, text="📥 Input Source", padding=10)
        source_frame.pack(fill='x', padx=5, pady=5)
        
        self.input_source_var = tk.StringVar(value="dataset")
        
        ttk.Radiobutton(source_frame, text="From Loaded Dataset", 
                       variable=self.input_source_var, value="dataset",
                       command=self.toggle_input_source).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="Paste Multiple Transcripts", 
                       variable=self.input_source_var, value="paste",
                       command=self.toggle_input_source).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="Load from CSV File", 
                       variable=self.input_source_var, value="file",
                       command=self.toggle_input_source).pack(anchor='w')
        
        # Dataset options frame
        self.dataset_options_frame = ttk.LabelFrame(left_frame, text="📊 Dataset Options", padding=10)
        self.dataset_options_frame.pack(fill='x', padx=5, pady=5)
        
        # Analysis type
        type_frame = ttk.Frame(self.dataset_options_frame)
        type_frame.pack(fill='x', pady=5)
        
        ttk.Label(type_frame, text="Group By:").pack(side='left')
        self.analysis_type_var = tk.StringVar(value="customer_type")
        
        type_combo = ttk.Combobox(type_frame, textvariable=self.analysis_type_var, width=20, state='readonly')
        type_combo['values'] = ('customer_type', 'city', 'customer_id')
        type_combo.pack(side='left', padx=10)
        type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_batch_combo())
        
        # Selection
        select_frame = ttk.Frame(self.dataset_options_frame)
        select_frame.pack(fill='x', pady=5)
        
        ttk.Label(select_frame, text="Select Value:").pack(side='left')
        self.batch_value_var = tk.StringVar()
        self.batch_value_combo = ttk.Combobox(select_frame, textvariable=self.batch_value_var, width=25)
        self.batch_value_combo.pack(side='left', padx=10)
        
        # Sample size
        sample_frame = ttk.Frame(self.dataset_options_frame)
        sample_frame.pack(fill='x', pady=5)
        
        ttk.Label(sample_frame, text="Sample Size:").pack(side='left')
        self.sample_size_var = tk.StringVar(value="20")
        sample_spin = ttk.Spinbox(sample_frame, from_=5, to=100, textvariable=self.sample_size_var, width=10)
        sample_spin.pack(side='left', padx=10)
        
        # Paste transcripts frame (initially hidden)
        self.paste_frame = ttk.LabelFrame(left_frame, text="📝 Paste Transcripts", padding=10)
        
        ttk.Label(self.paste_frame, text="Enter transcripts separated by '---' on a new line:").pack(anchor='w')
        
        self.batch_transcript_input = scrolledtext.ScrolledText(
            self.paste_frame,
            width=50,
            height=12,
            font=('Consolas', 9),
            bg='#16213e',
            fg='#eaeaea',
            insertbackground='white'
        )
        self.batch_transcript_input.pack(fill='both', expand=True, pady=5)
        self.batch_transcript_input.insert('1.0', """Transcript 1 content here...
---
Transcript 2 content here...
---
Transcript 3 content here...""")
        
        # File load frame (initially hidden)
        self.file_frame = ttk.LabelFrame(left_frame, text="📁 Load from File", padding=10)
        
        file_btn_frame = ttk.Frame(self.file_frame)
        file_btn_frame.pack(fill='x', pady=5)
        
        self.file_path_var = tk.StringVar(value="No file selected")
        ttk.Label(file_btn_frame, textvariable=self.file_path_var).pack(side='left')
        ttk.Button(file_btn_frame, text="Browse...", command=self.browse_transcript_file).pack(side='right')
        
        ttk.Label(self.file_frame, text="CSV should have a 'transcript' column").pack(anchor='w')
        
        # Run button frame
        run_frame = ttk.Frame(left_frame)
        run_frame.pack(fill='x', padx=5, pady=10)
        
        run_btn = ttk.Button(run_frame, text="🚀 Run Batch Analysis",
                            command=self.run_batch_analysis)
        run_btn.pack(pady=5)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var,
                                           maximum=100, length=300)
        self.progress_bar.pack(pady=5)
        
        self.progress_label = ttk.Label(run_frame, text="Ready")
        self.progress_label.pack()
        
        # Right pane - Results
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        results_frame = ttk.LabelFrame(right_frame, text="📊 Batch Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.batch_result_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 10),
            bg='#16213e',
            fg='#eaeaea'
        )
        self.batch_result_text.pack(fill='both', expand=True)
    
    def toggle_input_source(self):
        """Toggle between input source options"""
        source = self.input_source_var.get()
        
        # Hide all optional frames
        try:
            self.paste_frame.pack_forget()
        except:
            pass
        try:
            self.file_frame.pack_forget()
        except:
            pass
        try:
            self.dataset_options_frame.pack_forget()
        except:
            pass
        
        # Show appropriate frame - find the parent's children to place after source_frame
        parent = self.dataset_options_frame.master
        
        if source == "dataset":
            self.dataset_options_frame.pack(fill='x', padx=5, pady=5)
        elif source == "paste":
            self.paste_frame.pack(fill='both', expand=True, padx=5, pady=5)
        elif source == "file":
            self.file_frame.pack(fill='x', padx=5, pady=5)
    
    def browse_transcript_file(self):
        """Browse for a transcript file"""
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.file_path_var.set(filepath)
            self.loaded_file_path = filepath
    
    def create_results_tab(self):
        """Create results view tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📋 Saved Results")
        
        # List of saved results
        list_frame = ttk.Frame(tab)
        list_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        ttk.Label(list_frame, text="Saved Analysis Results:").pack()
        
        self.results_listbox = tk.Listbox(list_frame, width=40, height=25,
                                         bg='#16213e', fg='#eaeaea',
                                         selectbackground='#e94560')
        self.results_listbox.pack(fill='y', expand=True, pady=5)
        self.results_listbox.bind('<<ListboxSelect>>', self.load_saved_result)
        
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="🔄 Refresh", command=self.refresh_results_list).pack(side='left')
        ttk.Button(btn_frame, text="📂 Open Folder", command=self.open_output_folder).pack(side='left', padx=5)
        
        # Result viewer
        viewer_frame = ttk.LabelFrame(tab, text="Result Details", padding=10)
        viewer_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.result_viewer = scrolledtext.ScrolledText(
            viewer_frame,
            font=('Consolas', 10),
            bg='#16213e',
            fg='#eaeaea'
        )
        self.result_viewer.pack(fill='both', expand=True)
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=20, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='SubHeader.TLabel')
        self.status_label.pack(side='left')
        
        # Export button
        export_btn = ttk.Button(status_frame, text="💾 Export Results",
                               command=self.export_results)
        export_btn.pack(side='right')
    
    def clear_placeholder(self, event):
        """Clear placeholder text on focus"""
        if self.transcript_input.get('1.0', 'end-1c') == 'Paste your transcript here...':
            self.transcript_input.delete('1.0', 'end')
    
    def load_data(self):
        """Load the dataset"""
        self.update_status("Loading data...")
        
        try:
            paths = ["Data Voice Hackathon_Master.xlsx", "data/Data Voice Hackathon_Master.xlsx"]
            for path in paths:
                if os.path.exists(path):
                    self.df = pd.read_excel(path)
                    break
            
            if self.df is not None:
                self.data_label.config(text=f"📊 Loaded {len(self.df):,} records")
                self.update_status(f"Data loaded: {len(self.df):,} records")
                
                # Initialize agents
                self.insights_agent = InsightsAgent(verbose=False)
                self.aggregation_agent = AggregationAgent(verbose=False)
                
                # Update combo values
                self.update_batch_combo()
                
                # Refresh results list
                self.refresh_results_list()
            else:
                self.data_label.config(text="❌ Data not found")
                self.update_status("Error: Could not load data file")
                
        except Exception as e:
            self.data_label.config(text=f"❌ Error: {str(e)}")
            self.update_status(f"Error: {str(e)}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_batch_combo(self, *args):
        """Update batch analysis combo based on type"""
        if self.df is None:
            return
        
        analysis_type = self.analysis_type_var.get()
        
        if analysis_type == "customer_type":
            values = list(self.df['customer_type'].value_counts().head(20).index)
        elif analysis_type == "city":
            values = list(self.df['city_name'].value_counts().head(50).index)
        elif analysis_type == "customer_id":
            values = list(self.df['glid'].value_counts().head(50).index)
        else:
            values = []
        
        self.batch_value_combo['values'] = values
        if values:
            self.batch_value_var.set(values[0])
    
    def analyze_single_transcript(self):
        """Analyze single transcript"""
        transcript = self.transcript_input.get('1.0', 'end-1c').strip()
        
        if not transcript or transcript == 'Paste your transcript here...':
            messagebox.showwarning("Warning", "Please enter a transcript to analyze")
            return
        
        if self.insights_agent is None:
            messagebox.showerror("Error", "Agent not initialized. Please wait for data to load.")
            return
        
        self.update_status("Analyzing transcript...")
        self.single_result_text.delete('1.0', 'end')
        self.single_result_text.insert('end', "🔄 Analyzing...\n\n")
        self.root.update_idletasks()
        
        # Run in thread
        def analyze():
            metadata = {
                'customer_type': self.cust_type_var.get(),
                'city': self.city_var.get()
            }
            
            try:
                result = self.insights_agent.analyze_transcript(transcript, metadata)
                self.current_result = result
                self.current_transcript = transcript
                
                # Display results
                self.root.after(0, lambda: self.display_single_result(result))
                self.root.after(0, lambda: self.update_status("Analysis complete"))
                
            except Exception as e:
                self.root.after(0, lambda: self.single_result_text.insert('end', f"\n❌ Error: {str(e)}"))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def display_single_result(self, result):
        """Display single analysis result"""
        self.single_result_text.delete('1.0', 'end')
        
        if result.get('analysis_success'):
            text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ANALYSIS RESULTS                               ║
╚══════════════════════════════════════════════════════════════════╝

🏷️  PRIMARY CATEGORY: {result.get('primary_category', 'N/A')}

📝 ISSUE SUMMARY:
   {result.get('issue_summary', 'N/A')}

😊 SENTIMENT: {result.get('sentiment', 'N/A')}
📈 SENTIMENT SHIFT: {result.get('sentiment_shift', 'N/A')}
⚠️  CHURN RISK: {result.get('churn_risk', 'N/A')}
🚨 URGENCY: {result.get('urgency', 'N/A')}
✅ RESOLUTION: {result.get('resolution_status', 'N/A')}

😟 CUSTOMER PAIN POINTS:
"""
            for pp in result.get('customer_pain_points', []):
                text += f"   • {pp}\n"
            
            exec_perf = result.get('executive_performance', {})
            text += f"""
👨‍💼 EXECUTIVE PERFORMANCE:
   Empathy: {'✅' if exec_perf.get('empathy_shown') else '❌'}
   Solution Offered: {'✅' if exec_perf.get('solution_offered') else '❌'}
   Followed Process: {'✅' if exec_perf.get('followed_process') else '❌'}
   Escalation Needed: {'⚠️' if exec_perf.get('escalation_needed') else '✅ No'}

💡 ACTIONABLE INSIGHT:
   {result.get('actionable_insight', 'N/A')}

🔄 FOLLOW-UP REQUIRED: {'Yes - ' + result.get('follow_up_reason', '') if result.get('requires_follow_up') else 'No'}

🏷️  KEYWORDS: {', '.join(result.get('keywords', []))}
"""
        else:
            text = f"❌ Analysis failed: {result.get('error', 'Unknown error')}"
        
        self.single_result_text.insert('end', text)
    
    def ask_question(self):
        """Ask a follow-up question about the transcript"""
        question = self.question_var.get().strip()
        
        if not question:
            return
        
        if not hasattr(self, 'current_transcript'):
            messagebox.showwarning("Warning", "Please analyze a transcript first")
            return
        
        self.update_status("Processing question...")
        self.single_result_text.insert('end', f"\n\n❓ Question: {question}\n")
        self.single_result_text.insert('end', "🔄 Thinking...\n")
        self.root.update_idletasks()
        
        def ask():
            try:
                answer = self.insights_agent.ask_question(self.current_transcript, question)
                self.root.after(0, lambda: self.single_result_text.insert('end', f"\n💡 Answer:\n{answer}\n"))
                self.root.after(0, lambda: self.update_status("Question answered"))
            except Exception as e:
                self.root.after(0, lambda: self.single_result_text.insert('end', f"\n❌ Error: {str(e)}\n"))
        
        threading.Thread(target=ask, daemon=True).start()
        self.question_var.set("")
    
    def run_batch_analysis(self):
        """Run batch analysis from selected source"""
        if self.aggregation_agent is None:
            messagebox.showerror("Error", "Agent not initialized")
            return
        
        source = self.input_source_var.get()
        
        self.batch_result_text.delete('1.0', 'end')
        self.progress_var.set(0)
        
        if source == "dataset":
            self.run_dataset_analysis()
        elif source == "paste":
            self.run_paste_analysis()
        elif source == "file":
            self.run_file_analysis()
    
    def run_dataset_analysis(self):
        """Run analysis from loaded dataset"""
        if self.df is None:
            messagebox.showerror("Error", "Dataset not loaded")
            return
        
        analysis_type = self.analysis_type_var.get()
        value = self.batch_value_var.get()
        sample_size = int(self.sample_size_var.get())
        
        if not value:
            messagebox.showwarning("Warning", "Please select a value to analyze")
            return
        
        self.batch_result_text.insert('end', f"🔄 Starting {analysis_type} analysis for: {value}\n\n")
        self.update_status(f"Analyzing {analysis_type}: {value}")
        
        def run():
            try:
                self.aggregation_agent.verbose = False
                
                if analysis_type == "customer_type":
                    result = self.aggregation_agent.aggregate_by_customer_type(
                        self.df, value, sample_size=sample_size
                    )
                elif analysis_type == "city":
                    result = self.aggregation_agent.aggregate_by_location(self.df, value)
                elif analysis_type == "customer_id":
                    result = self.aggregation_agent.aggregate_by_customer(self.df, int(value))
                else:
                    result = {'error': 'Invalid analysis type'}
                
                self.current_result = result
                self.root.after(0, lambda: self.display_batch_result(result))
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.update_status("Batch analysis complete"))
                self.save_batch_result(result, analysis_type, value)
                
            except Exception as e:
                self.root.after(0, lambda: self.batch_result_text.insert('end', f"\n❌ Error: {str(e)}"))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def run_paste_analysis(self):
        """Run analysis from pasted transcripts"""
        text = self.batch_transcript_input.get('1.0', 'end-1c').strip()
        
        if not text or text.startswith("Transcript 1 content"):
            messagebox.showwarning("Warning", "Please paste transcripts to analyze")
            return
        
        # Split by separator
        transcripts_text = text.split('---')
        transcripts = []
        
        for i, t in enumerate(transcripts_text):
            t = t.strip()
            if t:
                transcripts.append({
                    'transcript': t,
                    'metadata': {
                        'customer_type': 'Unknown',
                        'city': 'Unknown',
                        'source': f'Pasted transcript {i+1}'
                    }
                })
        
        if not transcripts:
            messagebox.showwarning("Warning", "No valid transcripts found")
            return
        
        self.batch_result_text.insert('end', f"🔄 Analyzing {len(transcripts)} pasted transcripts...\n\n")
        self.update_status(f"Analyzing {len(transcripts)} transcripts")
        
        def run():
            try:
                self.aggregation_agent.verbose = False
                result = self.aggregation_agent.analyze_multiple_transcripts(transcripts, show_individual=False)
                
                # Generate summary
                summary = self.aggregation_agent.generate_executive_summary(result)
                result['executive_summary'] = summary
                
                self.current_result = result
                self.root.after(0, lambda: self.display_batch_result(result))
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.update_status("Batch analysis complete"))
                self.save_batch_result(result, "pasted", f"{len(transcripts)}_transcripts")
                
            except Exception as e:
                self.root.after(0, lambda: self.batch_result_text.insert('end', f"\n❌ Error: {str(e)}"))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def run_file_analysis(self):
        """Run analysis from loaded file"""
        if not hasattr(self, 'loaded_file_path') or not self.loaded_file_path:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        try:
            # Load file
            if self.loaded_file_path.endswith('.csv'):
                file_df = pd.read_csv(self.loaded_file_path)
            else:
                # Assume text file with transcripts separated by ---
                with open(self.loaded_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                transcripts_text = text.split('---')
                file_df = pd.DataFrame({'transcript': [t.strip() for t in transcripts_text if t.strip()]})
            
            # Check for transcript column
            transcript_col = None
            for col in file_df.columns:
                if 'transcript' in col.lower():
                    transcript_col = col
                    break
            
            if transcript_col is None:
                messagebox.showerror("Error", "No 'transcript' column found in file")
                return
            
            transcripts = []
            for i, row in file_df.iterrows():
                transcripts.append({
                    'transcript': row[transcript_col],
                    'metadata': {
                        'customer_type': row.get('customer_type', 'Unknown'),
                        'city': row.get('city_name', row.get('city', 'Unknown')),
                        'source': f'File row {i+1}'
                    }
                })
            
            self.batch_result_text.insert('end', f"🔄 Analyzing {len(transcripts)} transcripts from file...\n\n")
            self.update_status(f"Analyzing {len(transcripts)} transcripts from file")
            
            def run():
                try:
                    self.aggregation_agent.verbose = False
                    
                    # Limit to first 50 for performance
                    if len(transcripts) > 50:
                        self.root.after(0, lambda: self.batch_result_text.insert('end', 
                            f"⚠️ Limiting to first 50 transcripts for performance\n\n"))
                        analyze_transcripts = transcripts[:50]
                    else:
                        analyze_transcripts = transcripts
                    
                    result = self.aggregation_agent.analyze_multiple_transcripts(analyze_transcripts, show_individual=False)
                    
                    summary = self.aggregation_agent.generate_executive_summary(result)
                    result['executive_summary'] = summary
                    
                    self.current_result = result
                    self.root.after(0, lambda: self.display_batch_result(result))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.update_status("File analysis complete"))
                    self.save_batch_result(result, "file", os.path.basename(self.loaded_file_path))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.batch_result_text.insert('end', f"\n❌ Error: {str(e)}"))
                    self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
            
            threading.Thread(target=run, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def display_batch_result(self, result):
        """Display batch analysis result"""
        self.batch_result_text.delete('1.0', 'end')
        
        if 'error' in result:
            self.batch_result_text.insert('end', f"❌ {result['error']}")
            return
        
        agg = result.get('aggregated_insights', {})
        
        text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                   BATCH ANALYSIS RESULTS                          ║
╚══════════════════════════════════════════════════════════════════╝

📊 OVERVIEW
   Total Analyzed: {result.get('total_analyzed', 0)}
   Successful: {result.get('successful', 0)}

🏷️  CATEGORY DISTRIBUTION
"""
        for cat, count in agg.get('category_distribution', {}).items():
            pct = count / max(result.get('total_analyzed', 1), 1) * 100
            bar = "█" * int(pct / 5)
            text += f"   {cat:25} {bar} {count} ({pct:.1f}%)\n"
        
        text += "\n😊 SENTIMENT DISTRIBUTION\n"
        for sent, count in agg.get('sentiment_distribution', {}).items():
            text += f"   • {sent}: {count}\n"
        
        text += "\n⚠️  CHURN RISK DISTRIBUTION\n"
        for risk, count in agg.get('churn_risk_distribution', {}).items():
            text += f"   • {risk}: {count}\n"
        
        text += "\n😟 TOP PAIN POINTS\n"
        for pp, count in list(agg.get('top_pain_points', {}).items())[:5]:
            text += f"   • {pp}: {count}\n"
        
        exec_perf = agg.get('executive_performance', {})
        text += f"""
👨‍💼 EXECUTIVE PERFORMANCE
   Empathy Rate: {exec_perf.get('empathy_rate', 0)}%
   Solution Rate: {exec_perf.get('solution_rate', 0)}%
   Process Compliance: {exec_perf.get('process_compliance', 0)}%
   Escalation Rate: {exec_perf.get('escalation_rate', 0)}%
"""
        
        # Add recommendations if available
        for key in ['customer_recommendations', 'location_insights', 'segment_recommendations']:
            if key in result:
                text += f"\n{'=' * 60}\n💡 RECOMMENDATIONS\n{'=' * 60}\n\n"
                text += result[key]
        
        # Add executive summary if available
        if 'executive_summary' in result:
            text += f"\n{'=' * 60}\n📋 EXECUTIVE SUMMARY\n{'=' * 60}\n\n"
            text += result['executive_summary']
        
        self.batch_result_text.insert('end', text)
    
    def save_batch_result(self, result, analysis_type, value):
        """Save batch result to file"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Remove non-serializable data
        save_result = {k: v for k, v in result.items() if k != 'individual_results'}
        
        filename = f"{OUTPUT_DIR}/gui_analysis_{analysis_type}_{value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_result, f, ensure_ascii=False, indent=2, default=str)
        
        self.root.after(0, self.refresh_results_list)
    
    def refresh_results_list(self):
        """Refresh the list of saved results"""
        self.results_listbox.delete(0, 'end')
        
        if os.path.exists(OUTPUT_DIR):
            files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')], reverse=True)
            for f in files[:50]:  # Show last 50
                self.results_listbox.insert('end', f)
    
    def load_saved_result(self, event):
        """Load a saved result when selected"""
        selection = self.results_listbox.curselection()
        if not selection:
            return
        
        filename = self.results_listbox.get(selection[0])
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.result_viewer.delete('1.0', 'end')
            
            # Pretty print JSON
            data = json.loads(content)
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
            self.result_viewer.insert('end', pretty)
            
        except Exception as e:
            self.result_viewer.delete('1.0', 'end')
            self.result_viewer.insert('end', f"Error loading file: {str(e)}")
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.startfile(OUTPUT_DIR)
    
    def export_results(self):
        """Export current result to file"""
        if self.current_result is None:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=OUTPUT_DIR
        )
        
        if filepath:
            try:
                save_result = {k: v for k, v in self.current_result.items() 
                              if k != 'individual_results'}
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(save_result, f, ensure_ascii=False, indent=2, default=str)
                
                messagebox.showinfo("Success", f"Results exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")


def main():
    root = tk.Tk()
    app = InsightsEngineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

