import os
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import ollama
import tkinter as tk
from tkinter import ttk, messagebox

class MeetingMinutesApp:
    def __init__(self):
        self.model_whisper = whisper.load_model("base")
        self.sample_rate = 44100
        self.recording = []
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Meeting Minutes Recorder")
        self.root.geometry("400x300")
        
        # Meeting Title Entry
        self.title_frame = ttk.Frame(self.root, padding="10")
        self.title_frame.pack(fill=tk.X)
        ttk.Label(self.title_frame, text="Meeting Title:").pack(side=tk.LEFT)
        self.title_entry = ttk.Entry(self.title_frame)
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Recording Controls
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X)
        self.record_button = ttk.Button(self.control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(fill=tk.X)
        
        # Status Label
        self.status_var = tk.StringVar(value="Ready to record")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(pady=10)
        
        self.is_recording = False

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        meeting_title = self.title_entry.get().strip()
        if not meeting_title:
            messagebox.showerror("Error", "Please enter a meeting title")
            return
            
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.status_var.set("Recording...")
        self.recording = []
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.recording.append(indata.copy())
            
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=callback
        )
        self.stream.start()

    def stop_recording(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.status_var.set("Processing recording...")
        self.root.update()
        
        self.process_recording()

    def process_recording(self):
        try:
            # Create folder
            meeting_title = self.title_entry.get().strip()
            date_str = datetime.now().strftime("%Y-%m-%d")
            folder_name = f"{meeting_title}_{date_str}"
            os.makedirs(folder_name, exist_ok=True)
            
            # Save audio
            audio_path = os.path.join(folder_name, "recording.wav")
            audio_data = np.concatenate(self.recording, axis=0)
            sf.write(audio_path, audio_data, self.sample_rate)
            
            # Transcribe
            self.status_var.set("Transcribing audio...")
            self.root.update()
            result = self.model_whisper.transcribe(audio_path)
            transcript = result["text"]
            
            # Save transcript
            transcript_path = os.path.join(folder_name, "transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            # Generate minutes using Ollama
            self.status_var.set("Generating minutes...")
            self.root.update()
            
            prompt = f"""Please create meeting minutes from the following transcript. 
            Format the minutes with these sections:
            - Date and Time
            - Attendees (extract names mentioned)
            - Agenda Items
            - Key Decisions
            - Action Items (with assignees if mentioned)
            - Next Steps
            
            Transcript:
            {transcript}"""
            
            response = ollama.chat(model='mistral', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            # Save minutes
            minutes_path = os.path.join(folder_name, "minutes.txt")
            with open(minutes_path, "w", encoding="utf-8") as f:
                f.write(response['message']['content'])
            
            self.status_var.set("Done! Files saved in: " + folder_name)
            messagebox.showinfo("Success", f"Meeting minutes have been generated and saved in: {folder_name}")
            
        except Exception as e:
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MeetingMinutesApp()
    app.run()
