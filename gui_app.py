"""
IndiaMART Insights Engine - GUI Application
Simple Python GUI using Tkinter for transcript analysis

Features:
- Input: Text transcript OR Audio file (Vosk STT)
- Analysis: NVIDIA NIM LLM insights extraction
- Batch: Multiple transcripts or audio files

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
        self.root.geometry("1200x850")
        self.root.configure(bg='#1a1a2e')
        
        # Data
        self.df = None
        self.insights_agent = None
        self.aggregation_agent = None
        self.current_result = None
        self.vosk_stt = None  # Will be loaded on demand
        
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
        """Create single transcript/audio analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📝 Single Analysis")
        
        # Left panel - Input
        left_frame = ttk.LabelFrame(tab, text="Input (Text or Audio)", padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Input type selection
        input_type_frame = ttk.Frame(left_frame)
        input_type_frame.pack(fill='x', pady=5)
        
        self.input_type_var = tk.StringVar(value="text")
        
        ttk.Radiobutton(input_type_frame, text="📝 Text Transcript", 
                       variable=self.input_type_var, value="text",
                       command=self.toggle_input_type).pack(side='left', padx=10)
        ttk.Radiobutton(input_type_frame, text="🎤 Audio File", 
                       variable=self.input_type_var, value="audio",
                       command=self.toggle_input_type).pack(side='left', padx=10)
        
        # Text input frame
        self.text_input_frame = ttk.Frame(left_frame)
        self.text_input_frame.pack(fill='both', expand=True, pady=5)
        
        self.transcript_input = scrolledtext.ScrolledText(
            self.text_input_frame, 
            width=50, 
            height=15,
            font=('Consolas', 10),
            bg='#16213e',
            fg='#eaeaea',
            insertbackground='white'
        )
        self.transcript_input.pack(fill='both', expand=True)
        self.transcript_input.insert('1.0', 'Paste your transcript here...')
        self.transcript_input.bind('<FocusIn>', self.clear_placeholder)
        
        # Audio input frame (initially hidden)
        self.audio_input_frame = ttk.Frame(left_frame)
        
        # Audio file selection
        audio_file_frame = ttk.Frame(self.audio_input_frame)
        audio_file_frame.pack(fill='x', pady=10)
        
        self.audio_path_var = tk.StringVar(value="No audio file selected")
        ttk.Label(audio_file_frame, text="🎤 Audio File:").pack(side='left')
        ttk.Label(audio_file_frame, textvariable=self.audio_path_var, 
                 foreground='#a0a0a0').pack(side='left', padx=10)
        ttk.Button(audio_file_frame, text="Browse...", 
                  command=self.browse_audio_file).pack(side='right')
        
        # Audio info
        self.audio_info_label = ttk.Label(self.audio_input_frame, 
                                         text="Supported formats: MP3, WAV, M4A, OGG",
                                         foreground='#a0a0a0')
        self.audio_info_label.pack(anchor='w', pady=5)
        
        # STT status
        self.stt_status_label = ttk.Label(self.audio_input_frame, 
                                         text="🔊 Vosk STT: Ready (vosk-model-hi-0.22)",
                                         foreground='#00d26a')
        self.stt_status_label.pack(anchor='w', pady=5)
        
        # Transcribed text preview
        ttk.Label(self.audio_input_frame, text="📝 Transcribed Text (Preview):").pack(anchor='w', pady=(10, 0))
        self.transcribed_preview = scrolledtext.ScrolledText(
            self.audio_input_frame,
            width=50,
            height=10,
            font=('Consolas', 9),
            bg='#16213e',
            fg='#eaeaea',
            state='disabled'
        )
        self.transcribed_preview.pack(fill='both', expand=True, pady=5)
        
        # Transcribe button
        self.transcribe_btn = ttk.Button(self.audio_input_frame, text="🎤 Transcribe Audio",
                                        command=self.transcribe_audio)
        self.transcribe_btn.pack(pady=5)
        
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
        analyze_btn = ttk.Button(left_frame, text="🔍 Analyze with LLM", 
                                command=self.analyze_input)
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
    
    def toggle_input_type(self):
        """Toggle between text and audio input"""
        input_type = self.input_type_var.get()
        
        if input_type == "text":
            self.audio_input_frame.pack_forget()
            self.text_input_frame.pack(fill='both', expand=True, pady=5)
        else:
            self.text_input_frame.pack_forget()
            self.audio_input_frame.pack(fill='both', expand=True, pady=5)
    
    def browse_audio_file(self):
        """Browse for audio file"""
        filepath = filedialog.askopenfilename(
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("MP3 files", "*.mp3"),
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.audio_path_var.set(os.path.basename(filepath))
            self.selected_audio_path = filepath
            self.update_status(f"Audio file selected: {os.path.basename(filepath)}")
    
    def get_vosk_stt(self):
        """Initialize Vosk STT on demand"""
        if self.vosk_stt is None:
            try:
                from src.stt import VoskSTT
                self.stt_status_label.config(text="⏳ Loading Vosk model...", foreground='#ffc107')
                self.root.update_idletasks()
                
                self.vosk_stt = VoskSTT(model_path="vosk-model-hi-0.22", verbose=False)
                self.stt_status_label.config(text="✅ Vosk STT: Ready", foreground='#00d26a')
            except Exception as e:
                self.stt_status_label.config(text=f"❌ Vosk error: {str(e)}", foreground='#dc3545')
                return None
        return self.vosk_stt
    
    def transcribe_audio(self):
        """Transcribe audio file using Vosk"""
        if not hasattr(self, 'selected_audio_path') or not self.selected_audio_path:
            messagebox.showwarning("Warning", "Please select an audio file first")
            return
        
        self.update_status("Transcribing audio...")
        self.transcribe_btn.config(state='disabled')
        
        # Enable and clear preview
        self.transcribed_preview.config(state='normal')
        self.transcribed_preview.delete('1.0', 'end')
        self.transcribed_preview.insert('end', "🔄 Transcribing audio...\n\nPlease wait, this may take a moment.")
        self.transcribed_preview.config(state='disabled')
        self.root.update_idletasks()
        
        def transcribe():
            try:
                stt = self.get_vosk_stt()
                if stt is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load Vosk STT"))
                    return
                
                result = stt.transcribe(self.selected_audio_path)
                transcript = result.get('transcript', '')
                duration = result.get('duration', 0)
                
                self.current_transcript = transcript
                
                # Update preview
                def update_preview():
                    self.transcribed_preview.config(state='normal')
                    self.transcribed_preview.delete('1.0', 'end')
                    self.transcribed_preview.insert('end', f"Duration: {duration:.1f}s\n\n{transcript}")
                    self.transcribed_preview.config(state='disabled')
                    self.transcribe_btn.config(state='normal')
                    self.update_status(f"Transcription complete: {len(transcript)} characters")
                
                self.root.after(0, update_preview)
                
            except Exception as e:
                self.root.after(0, lambda: self.transcribed_preview.config(state='normal'))
                self.root.after(0, lambda: self.transcribed_preview.delete('1.0', 'end'))
                self.root.after(0, lambda: self.transcribed_preview.insert('end', f"❌ Error: {str(e)}"))
                self.root.after(0, lambda: self.transcribed_preview.config(state='disabled'))
                self.root.after(0, lambda: self.transcribe_btn.config(state='normal'))
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}"))
        
        threading.Thread(target=transcribe, daemon=True).start()
    
    def analyze_input(self):
        """Analyze either text or audio input"""
        input_type = self.input_type_var.get()
        
        if input_type == "text":
            self.analyze_text_transcript()
        else:
            self.analyze_audio_transcript()
    
    def analyze_text_transcript(self):
        """Analyze text transcript"""
        transcript = self.transcript_input.get('1.0', 'end-1c').strip()
        
        if not transcript or transcript == 'Paste your transcript here...':
            messagebox.showwarning("Warning", "Please enter a transcript to analyze")
            return
        
        self.current_transcript = transcript
        self.run_llm_analysis(transcript)
    
    def analyze_audio_transcript(self):
        """Analyze transcribed audio"""
        if not hasattr(self, 'current_transcript') or not self.current_transcript:
            messagebox.showwarning("Warning", "Please transcribe an audio file first")
            return
        
        self.run_llm_analysis(self.current_transcript)
    
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
