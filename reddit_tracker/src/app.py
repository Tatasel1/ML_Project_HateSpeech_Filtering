import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_loader import fetch_data, load_from_csv
from preprocessor import clean_text
from analyzer import analyze_sentiment, extract_topics
import threading
import os

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class RedditTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reddit Trend Tracker 🚀")
        self.geometry("1200x800")

        # Layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Reddit Tracker", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Data Source Selection
        self.source_label = ctk.CTkLabel(self.sidebar_frame, text="Data Source:", anchor="w")
        self.source_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.source_var = ctk.StringVar(value="API")
        self.api_radio = ctk.CTkRadioButton(self.sidebar_frame, text="Reddit API", variable=self.source_var, value="API", command=self.toggle_inputs)
        self.api_radio.grid(row=2, column=0, padx=20, pady=5, sticky="w")
        self.csv_radio = ctk.CTkRadioButton(self.sidebar_frame, text="Local CSV", variable=self.source_var, value="CSV", command=self.toggle_inputs)
        self.csv_radio.grid(row=3, column=0, padx=20, pady=5, sticky="w")

        # CSV File Selection
        self.browse_button = ctk.CTkButton(self.sidebar_frame, text="Browse CSV", command=self.browse_file)
        self.browse_button.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.file_label = ctk.CTkLabel(self.sidebar_frame, text="No file selected", anchor="w", font=("Arial", 10))
        self.file_label.grid(row=5, column=0, padx=20, pady=(0, 10))
        self.selected_file_path = None

        self.subreddits_label = ctk.CTkLabel(self.sidebar_frame, text="Subreddits (comma sep):", anchor="w")
        self.subreddits_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.subreddits_entry = ctk.CTkEntry(self.sidebar_frame)
        self.subreddits_entry.grid(row=7, column=0, padx=20, pady=(0, 10))
        self.subreddits_entry.insert(0, "technology,python,datascience")

        self.limit_label = ctk.CTkLabel(self.sidebar_frame, text="Max Posts:", anchor="w")
        self.limit_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.limit_slider = ctk.CTkSlider(self.sidebar_frame, from_=10, to=500, number_of_steps=49)
        self.limit_slider.grid(row=9, column=0, padx=20, pady=(0, 10))
        self.limit_slider.set(100)

        self.fetch_button = ctk.CTkButton(self.sidebar_frame, text="Fetch & Analyze", command=self.start_fetch_thread)
        self.fetch_button.grid(row=10, column=0, padx=20, pady=20)

        # Main Content Area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.status_label = ctk.CTkLabel(self.main_frame, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        # Tabs for Data and Visuals
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Data")
        self.tabview.add("Visualizations")
        self.tabview.add("Topics")

        # Data Tab
        self.data_textbox = ctk.CTkTextbox(self.tabview.tab("Data"), width=800)
        self.data_textbox.pack(fill="both", expand=True)

        # Visualizations Tab
        self.viz_frame = ctk.CTkFrame(self.tabview.tab("Visualizations"))
        self.viz_frame.pack(fill="both", expand=True)

        # Topics Tab
        self.topics_textbox = ctk.CTkTextbox(self.tabview.tab("Topics"), width=800)
        self.topics_textbox.pack(fill="both", expand=True)
        
        self.toggle_inputs() # Initial state

    def toggle_inputs(self):
        source = self.source_var.get()
        if source == "API":
            self.browse_button.configure(state="disabled")
            self.subreddits_entry.configure(state="normal")
            self.limit_slider.configure(state="normal")
        else:
            self.browse_button.configure(state="normal")
            self.subreddits_entry.configure(state="disabled")
            self.limit_slider.configure(state="disabled")

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.selected_file_path = filename
            self.file_label.configure(text=os.path.basename(filename))

    def start_fetch_thread(self):
        self.fetch_button.configure(state="disabled")
        self.status_label.configure(text="Fetching data... please wait.")
        threading.Thread(target=self.fetch_and_analyze, daemon=True).start()

    def fetch_and_analyze(self):
        try:
            source = self.source_var.get()
            
            if source == "API":
                subreddits = [s.strip() for s in self.subreddits_entry.get().split(",")]
                limit = int(self.limit_slider.get())
                df = fetch_data(subreddits, limit=limit)
            else:
                # Load from CSV
                if not self.selected_file_path:
                     self.update_status("Please select a CSV file.", error=True)
                     return
                df = load_from_csv(self.selected_file_path)
            
            if df.empty:
                self.update_status("No data found. Check API or CSV.", error=True)
                return

            # Validate columns
            # Normalize columns to lowercase and strip whitespace
            df.columns = df.columns.str.lower().str.strip()

            if 'title' not in df.columns:
                available_cols = ", ".join(df.columns.tolist())
                self.update_status(f"Error: Missing 'title' column. Found: {available_cols}", error=True)
                return
            
            if 'subreddit' not in df.columns:
                 df['subreddit'] = 'unknown' # Default if missing

            self.update_status("Analyzing text...")
            df['cleaned_text'] = df['title'].apply(clean_text)
            df = analyze_sentiment(df)

            # Update UI in main thread
            self.after(0, lambda: self.update_ui(df))

        except Exception as e:
            self.update_status(f"Error: {str(e)}", error=True)

    def update_status(self, message, error=False):
        color = "red" if error else "white" # Theme dependent, but simple for now
        self.after(0, lambda: self.status_label.configure(text=message))
        if error:
             self.after(0, lambda: self.fetch_button.configure(state="normal"))

    def update_ui(self, df):
        # 1. Show Data
        self.data_textbox.delete("0.0", "end")
        self.data_textbox.insert("0.0", df[['subreddit', 'title', 'score', 'sentiment']].head(20).to_string())

        # 2. Visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

        # Create Matplotlib Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Histogram
        ax1.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
        ax1.set_title('Sentiment Distribution')
        ax1.set_xlabel('Sentiment Score')
        
        # Bar Chart
        avg_sentiment = df.groupby("subreddit")['sentiment'].mean()
        avg_sentiment.plot(kind='bar', ax=ax2, color='salmon')
        ax2.set_title('Avg Sentiment by Subreddit')
        ax2.set_ylabel('Sentiment Score')
        plt.tight_layout()

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # 3. Topics
        topics = extract_topics(df)
        topic_str = ""
        for topic, words in topics.items():
            topic_str += f"{topic}: {', '.join(words)}\n\n"
        
        self.topics_textbox.delete("0.0", "end")
        self.topics_textbox.insert("0.0", topic_str)

        self.status_label.configure(text=f"Success! Fetched {len(df)} posts.")
        self.fetch_button.configure(state="normal")

if __name__ == "__main__":
    app = RedditTrackerApp()
    app.mainloop()

