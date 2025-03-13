import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
import threading
import sys
import csv
import json
from datetime import datetime
import io
import face_sample_collector
import face_database_manager
import recognition_system

class AttendanceSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("InsightFace Attendance System")
        self.root.geometry("800x600")
        
        # Create main tab control
        self.tab_control = ttk.Notebook(root)
        
        # Create tabs
        self.tab_registration = ttk.Frame(self.tab_control)
        self.tab_recognition = ttk.Frame(self.tab_control)
        self.tab_management = ttk.Frame(self.tab_control)
        self.tab_reports = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_registration, text='Registration')
        self.tab_control.add(self.tab_recognition, text='Recognition')
        self.tab_control.add(self.tab_management, text='Management')
        self.tab_control.add(self.tab_reports, text='Reports')
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Initialize tab contents
        self._init_registration_tab()
        self._init_recognition_tab()
        self._init_management_tab()
        self._init_reports_tab()
        
        # Initialize process tracking
        self.current_process = None
        
        # Initialize managers
        self.db_manager = face_database_manager.FaceDatabaseManager()
    
    def _init_registration_tab(self):
        """Initialize the Registration tab"""
        frame = ttk.Frame(self.tab_registration, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Person details frame
        details_frame = ttk.LabelFrame(frame, text="Person Details")
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Person ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.person_id_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.person_id_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Person Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.person_name_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.person_name_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Number of Samples:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.samples_var = tk.StringVar(value="5")
        samples_entry = ttk.Spinbox(details_frame, from_=1, to=10, textvariable=self.samples_var, width=5)
        samples_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Action buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=20)
        
        self.collect_btn = ttk.Button(buttons_frame, text="Collect Face Samples", command=self._collect_face_samples)
        self.collect_btn.pack(side=tk.LEFT, padx=5)
        
        self.register_btn = ttk.Button(buttons_frame, text="Register Person", command=self._register_person)
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(frame, text="Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_text.config(state=tk.DISABLED)
    
    def _init_recognition_tab(self):
        """Initialize the Recognition tab"""
        frame = ttk.Frame(self.tab_recognition, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Recognition settings
        settings_frame = ttk.LabelFrame(frame, text="Recognition Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.threshold_var = tk.StringVar(value="0.4")
        threshold_entry = ttk.Spinbox(settings_frame, from_=0.1, to=0.9, increment=0.1, textvariable=self.threshold_var, width=5)
        threshold_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.mark_attendance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Mark Attendance", variable=self.mark_attendance_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Action buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=20)
        
        self.start_recog_btn = ttk.Button(buttons_frame, text="Start Recognition", command=self._start_recognition)
        self.start_recog_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(frame, text="Recognition Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.recog_status_text = tk.Text(status_frame, height=10, wrap=tk.WORD)
        self.recog_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.recog_status_text.config(state=tk.DISABLED)
    
    def _init_management_tab(self):
        """Initialize the Management tab"""
        frame = ttk.Frame(self.tab_management, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Person list
        list_frame = ttk.LabelFrame(frame, text="Registered Persons")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create Treeview for person list
        self.person_tree = ttk.Treeview(list_frame, columns=("ID", "Name", "Samples", "Registration Date"), show="headings")
        self.person_tree.heading("ID", text="ID")
        self.person_tree.heading("Name", text="Name")
        self.person_tree.heading("Samples", text="Samples")
        self.person_tree.heading("Registration Date", text="Registration Date")
        
        self.person_tree.column("ID", width=100)
        self.person_tree.column("Name", width=200)
        self.person_tree.column("Samples", width=80)
        self.person_tree.column("Registration Date", width=120)
        
        self.person_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.person_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.person_tree.configure(yscrollcommand=scrollbar.set)
        
        # Management buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.refresh_btn = ttk.Button(button_frame, text="Refresh List", command=self._refresh_person_list)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        self.remove_btn = ttk.Button(button_frame, text="Remove Selected", command=self._remove_selected_person)
        self.remove_btn.pack(side=tk.LEFT, padx=5)
        
        # Load the initial list
        self._refresh_person_list()
    
    def _init_reports_tab(self):
        """Initialize the Reports tab"""
        frame = ttk.Frame(self.tab_reports, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Date selection
        date_frame = ttk.Frame(frame)
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(date_frame, text="Select Date:").pack(side=tk.LEFT, padx=5)
        
        self.report_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        date_entry = ttk.Entry(date_frame, textvariable=self.report_date_var, width=15)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        load_btn = ttk.Button(date_frame, text="Load Report", command=self._load_attendance_report)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Attendance report table
        report_frame = ttk.LabelFrame(frame, text="Attendance Report")
        report_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create Treeview for attendance data
        self.report_tree = ttk.Treeview(report_frame, columns=("ID", "Name", "Time", "Status"), show="headings")
        self.report_tree.heading("ID", text="Person ID")
        self.report_tree.heading("Name", text="Name")
        self.report_tree.heading("Time", text="Time")
        self.report_tree.heading("Status", text="Status")
        
        self.report_tree.column("ID", width=100)
        self.report_tree.column("Name", width=200)
        self.report_tree.column("Time", width=100)
        self.report_tree.column("Status", width=100)
        
        self.report_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        report_scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=self.report_tree.yview)
        report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_tree.configure(yscrollcommand=report_scrollbar.set)
        
        # Summary frame
        summary_frame = ttk.Frame(frame)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.total_var = tk.StringVar(value="Total: 0")
        self.present_var = tk.StringVar(value="Present: 0")
        self.absent_var = tk.StringVar(value="Absent: 0")
        
        ttk.Label(summary_frame, textvariable=self.total_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(summary_frame, textvariable=self.present_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(summary_frame, textvariable=self.absent_var).pack(side=tk.LEFT, padx=20)
    
    # Registration tab functions
    def _collect_face_samples(self):
        """Start face sample collection process"""
        person_id = self.person_id_var.get().strip()
        if not person_id:
            messagebox.showerror("Error", "Please enter a Person ID")
            return
        
        try:
            samples = int(self.samples_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of samples")
            return
        
        # Update status
        self._update_status(f"Starting face sample collection for {person_id}...\n")
        self._update_status("Please look at the camera and follow the instructions.\n")
        
        # Disable buttons during collection
        self.collect_btn.config(state=tk.DISABLED)
        self.register_btn.config(state=tk.DISABLED)
        
        # Run collection in a separate thread
        def collection_thread():
            collector = face_sample_collector.FaceSampleCollector(required_samples=samples)
            result = collector.collect_face_samples(person_id)
            
            # Re-enable buttons
            self.root.after(0, lambda: self.collect_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.register_btn.config(state=tk.NORMAL))
            
            if result:
                self.root.after(0, lambda: self._update_status(f"Successfully collected {samples} samples for {person_id}.\n"))
                self.root.after(0, lambda: self._update_status("You can now register this person using the 'Register Person' button.\n"))
            else:
                self.root.after(0, lambda: self._update_status("Sample collection interrupted or failed.\n"))
        
        thread = threading.Thread(target=collection_thread)
        thread.daemon = True
        thread.start()
    
    def _register_person(self):
        """Register a person in the database"""
        person_id = self.person_id_var.get().strip()
        person_name = self.person_name_var.get().strip()
        
        if not person_id:
            messagebox.showerror("Error", "Please enter a Person ID")
            return
        
        if not person_name:
            person_name = person_id  # Use ID as name if not provided
        
        # Update status
        self._update_status(f"Registering {person_id} ({person_name})...\n")
        
        # Register person
        result = self.db_manager.register_person(person_id, person_name)
        
        if result:
            self._update_status(f"Successfully registered {person_name} in the database.\n")
            # Refresh the person list
            self._refresh_person_list()
        else:
            self._update_status("Registration failed. Make sure you have collected face samples first.\n")
    
    def _update_status(self, message):
        """Update status text widget"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    # Recognition tab functions
    def _start_recognition(self):
        """Start face recognition process"""
        try:
            threshold = float(self.threshold_var.get())
            if threshold < 0.1 or threshold > 0.9:
                raise ValueError("Threshold must be between 0.1 and 0.9")
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold value")
            return
        
        mark_attendance = self.mark_attendance_var.get()
        
        # Update status
        self._update_recog_status("Starting face recognition...\n")
        self._update_recog_status(f"Threshold: {threshold}, Mark Attendance: {mark_attendance}\n")
        self._update_recog_status("Press 'q' in the recognition window to stop.\n")
        
        # Disable button during recognition
        self.start_recog_btn.config(state=tk.DISABLED)
        
        # Run recognition in a separate thread
        def recognition_thread():
            recognizer = recognition_system.FaceRecognitionSystem(threshold=threshold)
            recognizer.recognize_from_webcam(mark_attendance=mark_attendance)
            
            # Re-enable button
            self.root.after(0, lambda: self.start_recog_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self._update_recog_status("Recognition stopped.\n"))
            
            # Refresh reports tab if attendance was marked
            if mark_attendance:
                self.root.after(0, lambda: self._load_attendance_report())
        
        thread = threading.Thread(target=recognition_thread)
        thread.daemon = True
        thread.start()
    
    def _update_recog_status(self, message):
        """Update recognition status text widget"""
        self.recog_status_text.config(state=tk.NORMAL)
        self.recog_status_text.insert(tk.END, message)
        self.recog_status_text.see(tk.END)
        self.recog_status_text.config(state=tk.DISABLED)
    
    # Management tab functions
    def _refresh_person_list(self):
        """Refresh the person list in the management tab"""
        # Clear current items
        for item in self.person_tree.get_children():
            self.person_tree.delete(item)
        
        # Load metadata
        metadata_file = os.path.join('face_db', 'metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                    if "persons" in metadata:
                        for person_id, info in metadata["persons"].items():
                            self.person_tree.insert("", tk.END, values=(
                                person_id,
                                info.get("name", person_id),
                                info.get("sample_count", "?"),
                                info.get("registration_date", "Unknown")
                            ))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load metadata: {str(e)}")
    
    def _remove_selected_person(self):
        """Remove selected person from the database"""
        selection = self.person_tree.selection()
        if not selection:
            messagebox.showinfo("Information", "Please select a person to remove")
            return
        
        # Get the selected person's ID
        person_id = self.person_tree.item(selection[0], "values")[0]
        
        # Confirm removal
        confirmed = messagebox.askyesno("Confirm", f"Are you sure you want to remove {person_id}?")
        if not confirmed:
            return
        
        # Remove the person
        result = self.db_manager.remove_person(person_id)
        
        if result:
            messagebox.showinfo("Success", f"{person_id} has been removed from the database")
            self._refresh_person_list()
        else:
            messagebox.showerror("Error", f"Failed to remove {person_id}")
    
    # Reports tab functions
    def _load_attendance_report(self):
        """Load attendance data for the selected date"""
        date = self.report_date_var.get()
        attendance_file = f'attendance_{date}.csv'
        
        # Clear current items
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        if not os.path.exists(attendance_file):
            self.total_var.set("Total: 0")
            self.present_var.set("Present: 0")
            self.absent_var.set("Absent: 0")
            messagebox.showinfo("Information", f"No attendance data found for {date}")
            return
        
        try:
            # Read attendance data
            present_persons = set()
            with open(attendance_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header
                
                for row in reader:
                    if len(row) >= 4:
                        self.report_tree.insert("", tk.END, values=row)
                        present_persons.add(row[0])  # Add person ID to present set
            
            # Update statistics
            total_persons = len(self._get_all_registered_persons())
            present_count = len(present_persons)
            absent_count = total_persons - present_count
            
            self.total_var.set(f"Total: {total_persons}")
            self.present_var.set(f"Present: {present_count}")
            self.absent_var.set(f"Absent: {absent_count}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance data: {str(e)}")
    
    def _get_all_registered_persons(self):
        """Get all registered persons from metadata"""
        persons = set()
        metadata_file = os.path.join('face_db', 'metadata.json')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if "persons" in metadata:
                        for person_id in metadata["persons"]:
                            persons.add(person_id)
            except:
                pass
        
        return persons

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()