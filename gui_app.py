"""
IndiaMART Insights Engine - GUI Application
Desktop GUI using Tkinter for call translation and analysis

Features:
- Input: Call ID (click_to_call_id)
- Translation: Sarvam AI Translation API
- Analysis: NVIDIA NIM LLM insights extraction

Run with: python gui_app.py
"""

import os
import sys
import json
import threading
import requests
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import InsightsAgent, AggregationAgent
from src.config import NVIDIA_MODEL, OUTPUT_DIR
from src.ui.transcription_section import (
    fetch_call_by_id,
    transcribe_call,
    save_transcript_locally,
    load_cached_transcript
)

# Sarvam API Configuration
SARVAM_API_URL = "http://localhost:8888/translate"
TRANSCRIPTS_DIR = "output/transcripts"


class InsightsEngineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IndiaMART Insights Engine")
        self.root.geometry("1200x850")
        self.root.configure(bg='#faf8f3')  # Cream background
        
        # Data
        self.df = None
        self.insights_agent = None
        self.aggregation_agent = None
        self.current_result = None
        self.current_translation = None  # Store translation results
        self.current_call_data = None  # Store current call metadata
        
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
        
        # Colors - Purple and Cream theme
        self.colors = {
            'bg_cream': '#faf8f3',
            'bg_white': '#ffffff',
            'purple_dark': '#6b46c1',
            'purple_medium': '#805ad5',
            'text_dark': '#2d3748',
            'text_dim': '#718096',
            'success': '#00d26a',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
        
        style.configure('Header.TLabel', 
                       background=self.colors['bg_cream'], 
                       foreground=self.colors['purple_dark'],
                       font=('Segoe UI', 24, 'bold'))
        
        style.configure('SubHeader.TLabel',
                       background=self.colors['bg_cream'],
                       foreground=self.colors['text_dim'],
                       font=('Segoe UI', 10))
        
        style.configure('TButton',
                       font=('Segoe UI', 10),
                       padding=10,
                       relief='flat')
        
        style.configure('Accent.TButton',
                       background=self.colors['purple_dark'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat')
        
        style.configure('TNotebook',
                       background=self.colors['bg_cream'])
        
        style.configure('TNotebook.Tab',
                       font=('Segoe UI', 10),
                       padding=[15, 5])
        
        style.configure('TFrame',
                       background=self.colors['bg_cream'])
        
        style.configure('TLabelframe',
                       background=self.colors['bg_cream'],
                       foreground=self.colors['purple_dark'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('TLabelframe.Label',
                       background=self.colors['bg_cream'],
                       foreground=self.colors['purple_dark'],
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
                            text=f"Vosk STT + NVIDIA NIM ({NVIDIA_MODEL})",
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
        
        # Tab 1: Single Analysis (Text or Audio)
        self.create_single_analysis_tab()
        
        # Tab 2: Batch Analysis
        self.create_batch_analysis_tab()
        
        # Tab 3: Results View
        self.create_results_tab()
    
    def create_single_analysis_tab(self):
        """Create single call translation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Call Translation")
        
        # Left panel - Input
        left_frame = ttk.LabelFrame(tab, text="Call Translation Input", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Call ID input
        id_frame = ttk.Frame(left_frame)
        id_frame.pack(fill='x', pady=10)
        
        ttk.Label(id_frame, text="Call ID (click_to_call_id):").pack(anchor='w')
        self.call_id_var = tk.StringVar()
        call_id_entry = ttk.Entry(id_frame, textvariable=self.call_id_var, width=30, font=('Segoe UI', 11))
        call_id_entry.pack(fill='x', pady=5)
        call_id_entry.bind('<Return>', lambda e: self.fetch_and_translate())
        
        # Sarvam API status
        self.api_status_label = ttk.Label(left_frame, 
                                         text="Sarvam API: Checking...",
                                         foreground='#ffc107')
        self.api_status_label.pack(anchor='w', pady=5)
        
        # Check API status on startup
        self.root.after(500, self.check_api_status)
        
        # Translate button
        translate_btn = ttk.Button(left_frame, text="Fetch & Translate Call",
                                  command=self.fetch_and_translate,
                                  style='Accent.TButton')
        translate_btn.pack(pady=10)
        
        # Call metadata display
        meta_frame = ttk.LabelFrame(left_frame, text="Call Metadata", padding=10)
        meta_frame.pack(fill='x', pady=10)
        
        self.meta_text = scrolledtext.ScrolledText(
            meta_frame,
            width=40,
            height=8,
            font=('Segoe UI', 9),
            bg='#ffffff',
            fg='#2d3748',
            relief='solid',
            borderwidth=1,
            state='disabled'
        )
        self.meta_text.pack(fill='both', expand=True)
        
        # Translation preview
        preview_frame = ttk.LabelFrame(left_frame, text="Translation Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, pady=10)
        
        self.translation_preview = scrolledtext.ScrolledText(
            preview_frame,
            width=40,
            height=12,
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#2d3748',
            relief='solid',
            borderwidth=1,
            state='disabled',
            wrap='word'
        )
        self.translation_preview.pack(fill='both', expand=True)
        
        # Analyze translated text button
        analyze_btn = ttk.Button(left_frame, text="Analyze Translation with LLM",
                                command=self.analyze_translation)
        analyze_btn.pack(pady=5)
        
        # Right panel - Results
        right_frame = ttk.LabelFrame(tab, text="Analysis Results", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.single_result_text = scrolledtext.ScrolledText(
            right_frame,
            width=50,
            height=25,
            font=('Consolas', 10),
            bg='#f5f1e8',
            fg='#2d3748',
            relief='solid',
            borderwidth=1,
            wrap='word'
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
    
    
    def check_api_status(self):
        """Check if Sarvam API is accessible"""
        try:
            response = requests.get(SARVAM_API_URL.replace('/translate', '/health'), timeout=2)
            self.api_status_label.config(text="Sarvam API: ● Online", foreground='#00d26a')
        except:
            self.api_status_label.config(text="Sarvam API: ● Offline", foreground='#dc3545')
    
    def fetch_and_translate(self):
        """Fetch call by ID and translate using Sarvam API - replicates app.py logic"""
        call_id = self.call_id_var.get().strip()
        
        if not call_id:
            messagebox.showwarning("Warning", "Please enter a Call ID")
            return
        
        if self.df is None:
            messagebox.showerror("Error", "Dataset not loaded. Please wait for data to load.")
            return
        
        self.update_status(f"Fetching call {call_id}...")
        
        # Clear previous data
        self.meta_text.config(state='normal')
        self.meta_text.delete('1.0', 'end')
        self.meta_text.insert('end', "Fetching call data...")
        self.meta_text.config(state='disabled')
        
        self.translation_preview.config(state='normal')
        self.translation_preview.delete('1.0', 'end')
        self.translation_preview.insert('end', "Waiting for call data...")
        self.translation_preview.config(state='disabled')
        
        self.root.update_idletasks()
        
        def translate():
            try:
                # Fetch call from dataset using function from transcription_section.py
                call_row = fetch_call_by_id(self.df, call_id)
                
                if call_row is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"No call found with ID: {call_id}"))
                    self.root.after(0, lambda: self.update_status("Call not found"))
                    return
                
                self.current_call_data = call_row
                
                # Display metadata
                def show_metadata():
                    self.meta_text.config(state='normal')
                    self.meta_text.delete('1.0', 'end')
                    meta_info = f"""Call ID: {call_id}
Duration: {call_row.get('call_duration', 'N/A')}s
Direction: {call_row.get('FLAG_IN_OUT', 'N/A')}
City: {call_row.get('city_name', 'N/A')}
Customer Type: {call_row.get('customer_type', 'N/A')}
GLID: {call_row.get('glid', 'N/A')}
Date: {call_row.get('call_entered_on', 'N/A')}"""
                    self.meta_text.insert('end', meta_info)
                    self.meta_text.config(state='disabled')
                
                self.root.after(0, show_metadata)
                
                # Get audio URL
                audio_url = str(call_row.get('call_recording_url', ''))
                
                if not audio_url or audio_url == 'nan':
                    self.root.after(0, lambda: messagebox.showerror("Error", "No audio recording URL found for this call"))
                    self.root.after(0, lambda: self.update_status("No audio URL"))
                    return
                
                # Update preview
                def show_translating():
                    self.translation_preview.config(state='normal')
                    self.translation_preview.delete('1.0', 'end')
                    self.translation_preview.insert('end', "Translating audio... This may take a few seconds")
                    self.translation_preview.config(state='disabled')
                
                self.root.after(0, show_translating)
                self.root.after(0, lambda: self.update_status("Translating audio..."))
                
                # Check cache first
                cached_transcript = load_cached_transcript(
                    str(call_row.get('glid', '')),
                    call_id
                )
                
                if cached_transcript:
                    transcript_data = cached_transcript
                    self.root.after(0, lambda: self.update_status("Using cached translation"))
                else:
                    # Call Sarvam API using function from transcription_section.py
                    transcript_data = transcribe_call(audio_url)
                    
                    # Save to cache if successful
                    if transcript_data.get('status') == 'success':
                        save_transcript_locally(
                            str(call_row.get('glid', '')),
                            str(call_row.get('call_entered_on', '')),
                            call_id,
                            transcript_data
                        )
                
                self.current_translation = transcript_data
                
                # Display translation
                def show_translation():
                    self.translation_preview.config(state='normal')
                    self.translation_preview.delete('1.0', 'end')
                    
                    if transcript_data.get('status') == 'success':
                        language_code = transcript_data.get('language_code', 'Unknown')
                        self.translation_preview.insert('end', f"Language: {language_code}\\n\\n")
                        
                        speakers = transcript_data.get('speakers', [])
                        if speakers:
                            for speaker in speakers:
                                speaker_id = speaker.get('speaker_id', 'Unknown')
                                text = speaker.get('text', '')
                                self.translation_preview.insert('end', f"{speaker_id}:\\n{text}\\n\\n")
                        else:
                            self.translation_preview.insert('end', "No speaker data available")
                    else:
                        error_msg = transcript_data.get('message', 'Unknown error')
                        self.translation_preview.insert('end', f"Translation Failed:\\n{error_msg}")
                    
                    self.translation_preview.config(state='disabled')
                    self.update_status("Translation complete")
                
                self.root.after(0, show_translation)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Translation error: {str(e)}"))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
        
        threading.Thread(target=translate, daemon=True).start()
    
    def analyze_translation(self):
        """Analyze the translated text with NVIDIA NIM"""
        if self.current_translation is None:
            messagebox.showwarning("Warning", "Please translate a call first")
            return
        
        if self.current_translation.get('status') != 'success':
            messagebox.showerror("Error", "Translation failed. Cannot analyze.")
            return
        
        # Extract full transcript text
        speakers = self.current_translation.get('speakers', [])
        if not speakers:
            messagebox.showwarning("Warning", "No transcript data to analyze")
            return
        
        # Combine all speaker text
        transcript_lines = []
        for speaker in speakers:
            speaker_id = speaker.get('speaker_id', 'Unknown')
            text = speaker.get('text', '')
            transcript_lines.append(f"{speaker_id}: {text}")
        
        full_transcript = "\\n".join(transcript_lines)
        
        # Run LLM analysis
        self.run_llm_analysis(full_transcript)
    
    def run_llm_analysis(self, transcript):
        """Run LLM analysis on transcript"""
        if self.insights_agent is None:
            messagebox.showerror("Error", "Agent not initialized. Please wait for data to load.")
            return
        
        self.update_status("Analyzing with LLM...")
        self.single_result_text.delete('1.0', 'end')
        self.single_result_text.insert('end', "🔄 Analyzing transcript with NVIDIA NIM...\n\n")
        self.root.update_idletasks()
        
        def analyze():
            metadata = {
                'customer_type': self.cust_type_var.get(),
                'city': self.city_var.get()
            }
            
            try:
                result = self.insights_agent.analyze_transcript(transcript, metadata)
                self.current_result = result
                
                self.root.after(0, lambda: self.display_single_result(result))
                self.root.after(0, lambda: self.update_status("Analysis complete"))
                
            except Exception as e:
                self.root.after(0, lambda: self.single_result_text.insert('end', f"\n❌ Error: {str(e)}"))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
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
        
        self.batch_source_var = tk.StringVar(value="dataset")
        
        ttk.Radiobutton(source_frame, text="From Loaded Dataset", 
                       variable=self.batch_source_var, value="dataset",
                       command=self.toggle_batch_source).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="Paste Multiple Transcripts", 
                       variable=self.batch_source_var, value="paste",
                       command=self.toggle_batch_source).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="Load from CSV File", 
                       variable=self.batch_source_var, value="file",
                       command=self.toggle_batch_source).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="🎤 Audio Folder", 
                       variable=self.batch_source_var, value="audio_folder",
                       command=self.toggle_batch_source).pack(anchor='w')
        
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
        
        # Audio folder frame (initially hidden)
        self.audio_folder_frame = ttk.LabelFrame(left_frame, text="🎤 Audio Folder", padding=10)
        
        audio_folder_btn_frame = ttk.Frame(self.audio_folder_frame)
        audio_folder_btn_frame.pack(fill='x', pady=5)
        
        self.audio_folder_var = tk.StringVar(value="No folder selected")
        ttk.Label(audio_folder_btn_frame, textvariable=self.audio_folder_var).pack(side='left')
        ttk.Button(audio_folder_btn_frame, text="Browse...", command=self.browse_audio_folder).pack(side='right')
        
        ttk.Label(self.audio_folder_frame, text="Process all .mp3/.wav files in folder").pack(anchor='w')
        
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
    
    def toggle_batch_source(self):
        """Toggle between batch input source options"""
        source = self.batch_source_var.get()
        
        # Hide all optional frames
        for frame in [self.paste_frame, self.file_frame, self.dataset_options_frame, self.audio_folder_frame]:
            try:
                frame.pack_forget()
            except:
                pass
        
        # Show appropriate frame
        if source == "dataset":
            self.dataset_options_frame.pack(fill='x', padx=5, pady=5)
        elif source == "paste":
            self.paste_frame.pack(fill='both', expand=True, padx=5, pady=5)
        elif source == "file":
            self.file_frame.pack(fill='x', padx=5, pady=5)
        elif source == "audio_folder":
            self.audio_folder_frame.pack(fill='x', padx=5, pady=5)
    
    def browse_audio_folder(self):
        """Browse for audio folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.audio_folder_var.set(folder)
            self.selected_audio_folder = folder
            
            # Count audio files
            audio_files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.wav', '.m4a', '.ogg'))]
            self.update_status(f"Found {len(audio_files)} audio files in folder")
    
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
            self.file_path_var.set(os.path.basename(filepath))
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
            paths = ["Data Voice Hackathon_Master-1.xlsx", "data/Data Voice Hackathon_Master-1.xlsx"]
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
                self.data_label.config(text="⚠️ No dataset (standalone mode)")
                self.update_status("Running in standalone mode")
                
                # Still initialize agents
                self.insights_agent = InsightsAgent(verbose=False)
                self.aggregation_agent = AggregationAgent(verbose=False)
                
        except Exception as e:
            self.data_label.config(text=f"⚠️ Standalone mode")
            self.update_status(f"Standalone mode: {str(e)}")
            
            # Initialize agents anyway
            try:
                self.insights_agent = InsightsAgent(verbose=False)
                self.aggregation_agent = AggregationAgent(verbose=False)
            except:
                pass
    
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
    
    def display_single_result(self, result):
        """Display single analysis result"""
        self.single_result_text.delete('1.0', 'end')
        
        if result.get('analysis_success'):
            text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ANALYSIS RESULTS                               ║
╚══════════════════════════════════════════════════════════════════╝

🏷️  PRIMARY CATEGORY: {result.get('primary_category', 'N/A')}
🎭 SELLER UNDERTONE: {result.get('seller_undertone', 'N/A')}

📝 ISSUE SUMMARY:
   {result.get('issue_summary', 'N/A')}

⚠️  CHURN RISK: {result.get('churn_risk_assessment', {}).get('risk_level', 'N/A')}

"""
            # Pain points
            pain_points = result.get('seller_pain_points', {})
            if pain_points:
                text += "😟 SELLER PAIN POINTS:\n"
                for k, v in pain_points.items():
                    if v and v != 'N/A' and v != 'None':
                        text += f"   • {k}: {v}\n"
            
            # Opportunities
            opps = result.get('opportunities', {})
            if opps.get('upsell_opportunity'):
                text += f"\n💰 UPSELL OPPORTUNITY: {opps.get('upsell_type', 'Yes')}\n"
            
            # Education needed
            seller_understanding = result.get('seller_understanding', {})
            if seller_understanding.get('needs_base_education'):
                text += f"\n📚 NEEDS EDUCATION: {', '.join(seller_understanding.get('education_topics_needed', []))}\n"
            
            # Talking points
            talking_points = result.get('top_5_talking_points', [])
            if talking_points:
                text += "\n💡 TOP TALKING POINTS:\n"
                for i, point in enumerate(talking_points[:5], 1):
                    text += f"   {i}. {point}\n"
            
            # Recommendation
            text += f"\n🎯 RECOMMENDATION:\n   {result.get('proactive_recommendation', 'N/A')}\n"
            
            # Processing time
            text += f"\n⏱️ Processing time: {result.get('processing_time', 'N/A')}s"
        else:
            text = f"❌ Analysis failed: {result.get('error', 'Unknown error')}"
        
        self.single_result_text.insert('end', text)
    
    def ask_question(self):
        """Ask a follow-up question about the transcript"""
        question = self.question_var.get().strip()
        
        if not question:
            return
        
        if not hasattr(self, 'current_transcript') or not self.current_transcript:
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
        source = self.batch_source_var.get()
        
        self.batch_result_text.delete('1.0', 'end')
        self.progress_var.set(0)
        
        if source == "dataset":
            self.run_dataset_analysis()
        elif source == "paste":
            self.run_paste_analysis()
        elif source == "file":
            self.run_file_analysis()
        elif source == "audio_folder":
            self.run_audio_folder_analysis()
    
    def run_audio_folder_analysis(self):
        """Process audio files from folder"""
        if not hasattr(self, 'selected_audio_folder') or not self.selected_audio_folder:
            messagebox.showwarning("Warning", "Please select an audio folder first")
            return
        
        folder = self.selected_audio_folder
        audio_files = [os.path.join(folder, f) for f in os.listdir(folder) 
                      if f.endswith(('.mp3', '.wav', '.m4a', '.ogg'))]
        
        if not audio_files:
            messagebox.showwarning("Warning", "No audio files found in folder")
            return
        
        self.batch_result_text.insert('end', f"🎤 Processing {len(audio_files)} audio files...\n\n")
        self.update_status(f"Processing {len(audio_files)} audio files")
        
        def run():
            try:
                stt = self.get_vosk_stt()
                if stt is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load Vosk STT"))
                    return
                
                results = []
                total = len(audio_files)
                
                for i, audio_path in enumerate(audio_files, 1):
                    filename = os.path.basename(audio_path)
                    self.root.after(0, lambda i=i, f=filename: self.batch_result_text.insert('end', f"[{i}/{total}] Transcribing: {f}\n"))
                    self.root.after(0, lambda i=i: self.progress_var.set(i / total * 50))
                    
                    # Transcribe
                    try:
                        transcript_result = stt.transcribe(audio_path)
                        transcript = transcript_result.get('transcript', '')
                        
                        if transcript:
                            # Analyze with LLM
                            self.root.after(0, lambda i=i: self.batch_result_text.insert('end', f"    Analyzing with LLM...\n"))
                            
                            insights = self.insights_agent.analyze_transcript(transcript, {'source': filename})
                            
                            results.append({
                                'file': filename,
                                'transcript': transcript[:500],
                                'category': insights.get('primary_category', 'N/A'),
                                'undertone': insights.get('seller_undertone', 'N/A'),
                                'churn_risk': insights.get('churn_risk_assessment', {}).get('risk_level', 'N/A'),
                                'summary': insights.get('issue_summary', 'N/A')
                            })
                            
                            self.root.after(0, lambda cat=insights.get('primary_category'): 
                                self.batch_result_text.insert('end', f"    ✅ Category: {cat}\n\n"))
                    except Exception as e:
                        self.root.after(0, lambda f=filename, e=str(e): 
                            self.batch_result_text.insert('end', f"    ❌ Error: {e}\n\n"))
                    
                    self.root.after(0, lambda i=i: self.progress_var.set(50 + i / total * 50))
                
                # Display summary
                def show_summary():
                    self.batch_result_text.insert('end', "\n" + "="*60 + "\n")
                    self.batch_result_text.insert('end', "📊 BATCH SUMMARY\n")
                    self.batch_result_text.insert('end', "="*60 + "\n\n")
                    
                    self.batch_result_text.insert('end', f"Total files: {len(audio_files)}\n")
                    self.batch_result_text.insert('end', f"Processed: {len(results)}\n\n")
                    
                    # Category distribution
                    categories = {}
                    for r in results:
                        cat = r.get('category', 'N/A')
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    self.batch_result_text.insert('end', "Category Distribution:\n")
                    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                        self.batch_result_text.insert('end', f"  • {cat}: {count}\n")
                    
                    self.progress_var.set(100)
                    self.update_status("Audio batch analysis complete")
                
                self.root.after(0, show_summary)
                
                # Save results
                self.current_result = {'audio_results': results, 'total': len(audio_files)}
                self.save_batch_result(self.current_result, "audio_folder", os.path.basename(folder))
                
            except Exception as e:
                self.root.after(0, lambda: self.batch_result_text.insert('end', f"\n❌ Error: {str(e)}"))
        
        threading.Thread(target=run, daemon=True).start()
    
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
        
        transcripts_text = text.split('---')
        transcripts = []
        
        for i, t in enumerate(transcripts_text):
            t = t.strip()
            if t:
                transcripts.append({
                    'transcript': t,
                    'metadata': {'source': f'Pasted transcript {i+1}'}
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
                
                summary = self.aggregation_agent.generate_executive_summary(result)
                result['executive_summary'] = summary
                
                self.current_result = result
                self.root.after(0, lambda: self.display_batch_result(result))
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.update_status("Batch analysis complete"))
                self.save_batch_result(result, "pasted", f"{len(transcripts)}_transcripts")
                
            except Exception as e:
                self.root.after(0, lambda: self.batch_result_text.insert('end', f"\n❌ Error: {str(e)}"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def run_file_analysis(self):
        """Run analysis from loaded file"""
        if not hasattr(self, 'loaded_file_path') or not self.loaded_file_path:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        try:
            if self.loaded_file_path.endswith('.csv'):
                file_df = pd.read_csv(self.loaded_file_path)
            else:
                with open(self.loaded_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                transcripts_text = text.split('---')
                file_df = pd.DataFrame({'transcript': [t.strip() for t in transcripts_text if t.strip()]})
            
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
                    'metadata': {'source': f'File row {i+1}'}
                })
            
            self.batch_result_text.insert('end', f"🔄 Analyzing {len(transcripts)} transcripts from file...\n\n")
            
            def run():
                try:
                    self.aggregation_agent.verbose = False
                    
                    if len(transcripts) > 50:
                        self.root.after(0, lambda: self.batch_result_text.insert('end', 
                            f"⚠️ Limiting to first 50 transcripts\n\n"))
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
        
        if 'executive_summary' in result:
            text += f"\n{'=' * 60}\n📋 EXECUTIVE SUMMARY\n{'=' * 60}\n\n"
            text += result['executive_summary']
        
        self.batch_result_text.insert('end', text)
    
    def save_batch_result(self, result, analysis_type, value):
        """Save batch result to file"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
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
            for f in files[:50]:
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
