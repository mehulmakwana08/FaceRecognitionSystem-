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
import cv2
import face_sample_collector
import face_database_manager
import recognition_system
from PIL import Image, ImageTk  # Add PIL imports for image conversion

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
        
        # Initialize process tracking
        self.current_process = None
        
        # Variables for camera preview
        self.camera_active = False
        self.camera_cap = None
        self.preview_after_id = None
        
        # Use the same directory for both components
        self.samples_dir = 'face_samples'
        
        # Initialize the database manager first
        self.db_manager = None
        self._init_database_manager()
            
        # Initialize the face collector
        self.collector = face_sample_collector.FaceSampleCollector(save_dir=self.samples_dir)
        
        # Initialize tab contents
        self._init_registration_tab()
        self._init_recognition_tab()
        self._init_management_tab()
        self._init_reports_tab()
        
        # Add tab change event to manage camera resources
        self.tab_control.bind("<<NotebookTabChanged>>", self._on_tab_change)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Add recognition control flag
        self.stop_recognition = False
        self.recognition_active = False
    
    def _init_database_manager(self):
        """Initialize connection to the database"""
        try:
            self.db_manager = face_database_manager.FaceDatabaseManager(samples_dir=self.samples_dir)
            print("Database manager initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing database manager: {e}")
            messagebox.showerror("Database Connection Error", 
                               f"Failed to connect to Milvus database: {str(e)}\n\n"
                               "Please ensure that:\n"
                               "1. Milvus server is running\n"
                               "2. Connection settings are correct\n\n"
                               "You can retry the connection from the Management tab.")
            self.db_manager = None
            return False
    
    def _init_registration_tab(self):
        """Initialize the Registration tab"""
        frame = ttk.Frame(self.tab_registration, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Person details frame
        details_frame = ttk.LabelFrame(frame, text="Person Details")
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        ttk.Label(details_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.reg_camera_var = tk.StringVar(value="0")
        self.reg_camera_combobox = ttk.Combobox(details_frame, textvariable=self.reg_camera_var, width=5)
        self.reg_camera_combobox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add button to refresh camera list
        refresh_cam_btn = ttk.Button(details_frame, text="Refresh", command=self._refresh_reg_camera_list)
        refresh_cam_btn.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Registration Number:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.registration_number_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.registration_number_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Full Name:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.full_name_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.full_name_var, width=30).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Mobile Number:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.mobile_number_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.mobile_number_var, width=20).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(details_frame, text="Number of Samples:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.samples_var = tk.StringVar(value="5")
        samples_entry = ttk.Spinbox(details_frame, from_=1, to=10, textvariable=self.samples_var, width=5)
        samples_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Camera preview frame
        preview_frame = ttk.LabelFrame(frame, text="Camera Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for camera preview
        self.reg_preview_canvas = tk.Canvas(preview_frame, bg="black")
        self.reg_preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Action buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_preview_btn = ttk.Button(buttons_frame, text="Start Preview", command=self._start_reg_preview)
        self.start_preview_btn.pack(side=tk.LEFT, padx=5)
        
        self.collect_btn = ttk.Button(buttons_frame, text="Collect Face Samples", command=self._collect_face_samples)
        self.collect_btn.pack(side=tk.LEFT, padx=5)
        
        self.register_btn = ttk.Button(buttons_frame, text="Register Person", command=self._register_person)
        self.register_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(frame, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_text = tk.Text(status_frame, height=5, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_text.config(state=tk.DISABLED)
        
        # Initialize camera list for registration
        self._refresh_reg_camera_list()
    
    def _init_recognition_tab(self):
        """Initialize the Recognition tab"""
        frame = ttk.Frame(self.tab_recognition, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Recognition settings
        settings_frame = ttk.LabelFrame(frame, text="Recognition Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        ttk.Label(settings_frame, text="Camera:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.camera_var = tk.StringVar(value="0")
        self.camera_combobox = ttk.Combobox(settings_frame, textvariable=self.camera_var, width=5)
        self.camera_combobox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Add button to refresh camera list
        refresh_cam_btn = ttk.Button(settings_frame, text="Refresh", command=self._refresh_camera_list)
        refresh_cam_btn.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Similarity Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.threshold_var = tk.StringVar(value="0.4")
        threshold_entry = ttk.Spinbox(settings_frame, from_=0.1, to=0.9, increment=0.1, textvariable=self.threshold_var, width=5)
        threshold_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.mark_attendance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Mark Attendance", variable=self.mark_attendance_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Camera preview frame
        recog_preview_frame = ttk.LabelFrame(frame, text="Recognition Preview")
        recog_preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for camera preview
        self.recog_preview_canvas = tk.Canvas(recog_preview_frame, bg="black")
        self.recog_preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Action buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_recog_preview_btn = ttk.Button(buttons_frame, text="Start Preview", command=self._start_recog_preview)
        self.start_recog_preview_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_recog_btn = ttk.Button(buttons_frame, text="Start Recognition", command=self._toggle_recognition)
        self.start_recog_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(frame, text="Recognition Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.recog_status_text = tk.Text(status_frame, height=5, wrap=tk.WORD)
        self.recog_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.recog_status_text.config(state=tk.DISABLED)
        
        # Initialize camera list
        self._refresh_camera_list()
    
    # Also enhance the refresh method with better error handling
    def _refresh_reg_camera_list(self):
        """Refresh the list of available cameras for registration tab"""
        # Check available cameras (up to index 5 should be enough for most systems)
        available_cameras = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(str(i))
                cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
        
        # Update combobox values
        if available_cameras:
            self.reg_camera_combobox['values'] = available_cameras
            self.reg_camera_var.set(available_cameras[0])  # Set to first available camera
        else:
            self.reg_camera_combobox['values'] = ["0"]  # Default to camera 0
            self.reg_camera_var.set("0")
        
        # Update status
        self._update_status(f"Found {len(available_cameras)} camera(s)\n")
    
    # Also make the refresh method more robust
    def _refresh_camera_list(self):
        """Refresh the list of available cameras"""
        # Check available cameras (up to index 5 should be enough for most systems)
        available_cameras = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(str(i))
                cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {str(e)}")
        
        # Update combobox values
        if available_cameras:
            self.camera_combobox['values'] = available_cameras
            self.camera_var.set(available_cameras[0])  # Set to first available camera
        else:
            self.camera_combobox['values'] = ["0"]  # Default to camera 0
            self.camera_var.set("0")
        
        # Update status
        self._update_recog_status(f"Found {len(available_cameras)} camera(s)\n")
    
    def _init_management_tab(self):
        """Initialize the Management tab"""
        frame = ttk.Frame(self.tab_management, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Database connection status frame
        db_frame = ttk.LabelFrame(frame, text="Database Connection")
        db_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Connection status indicator
        self.db_status_var = tk.StringVar(value="Disconnected" if self.db_manager is None else "Connected")
        db_status_color = "red" if self.db_manager is None else "green"
        
        status_frame = ttk.Frame(db_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        db_status_label = ttk.Label(status_frame, textvariable=self.db_status_var, foreground=db_status_color)
        db_status_label.pack(side=tk.LEFT, padx=5)
        
        # Retry connection button
        retry_btn = ttk.Button(status_frame, text="Retry Connection", command=self._retry_db_connection)
        retry_btn.pack(side=tk.RIGHT, padx=5)
        
        # Person list
        list_frame = ttk.LabelFrame(frame, text="Registered Persons")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create Treeview for person list
        self.person_tree = ttk.Treeview(list_frame, columns=("ID", "Name", "Mobile", "Samples", "Registration Date"), show="headings")
        self.person_tree.heading("ID", text="ID")
        self.person_tree.heading("Name", text="Name")
        self.person_tree.heading("Mobile", text="Mobile Number")
        self.person_tree.heading("Samples", text="Samples")
        self.person_tree.heading("Registration Date", text="Registration Date")
        
        self.person_tree.column("ID", width=100)
        self.person_tree.column("Name", width=150)
        self.person_tree.column("Mobile", width=120)
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

    def _retry_db_connection(self):
        """Attempt to reconnect to the database"""
        if self._init_database_manager():
            self.db_status_var.set("Connected")
            for widget in self.tab_management.winfo_children():
                for label in widget.winfo_children():
                    if isinstance(label, ttk.Label) and label.cget("textvariable") == str(self.db_status_var):
                        label.config(foreground="green")
            messagebox.showinfo("Connection Successful", "Successfully connected to database.")
            self._refresh_person_list()
        else:
            self.db_status_var.set("Disconnected")
            for widget in self.tab_management.winfo_children():
                for label in widget.winfo_children():
                    if isinstance(label, ttk.Label) and label.cget("textvariable") == str(self.db_status_var):
                        label.config(foreground="red")
    
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
        registration_number = self.registration_number_var.get().strip()
        if not registration_number:
            messagebox.showerror("Error", "Please enter a Registration Number")
            return
        
        try:
            samples = int(self.samples_var.get())
            camera_index = int(self.reg_camera_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
            return
        
        # Update status
        self._update_status(f"Starting face sample collection for {registration_number}...\n")
        self._update_status(f"Using camera {camera_index}\n")
        self._update_status("Please look at the camera and follow the instructions.\n")
        
        # Disable buttons during collection
        self.collect_btn.config(state=tk.DISABLED)
        self.register_btn.config(state=tk.DISABLED)
        self.start_preview_btn.config(state=tk.DISABLED)
        
        # Stop any ongoing preview
        self._stop_camera()
        
        # Run collection in a separate thread
        def collection_thread():
            try:
                success = self.collector.collect_face_samples(
                    registration_number, 
                    camera_index=camera_index,
                    preview_widget=self.reg_preview_canvas,
                    convert_func=self._convert_cv_to_tkimage,
                    app_root=self.root
                )
                
                # Re-enable buttons
                self.root.after(0, lambda: self.collect_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.register_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.start_preview_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.start_preview_btn.config(text="Start Preview"))
                
                if success:
                    self.root.after(0, lambda: self._update_status(f"Successfully collected {samples} samples for {registration_number}.\n"))
                    self.root.after(0, lambda: self._update_status("You can now register this person using the 'Register Person' button.\n"))
                else:
                    self.root.after(0, lambda: self._update_status("Sample collection interrupted or failed.\n"))
            except Exception as e:
                self.root.after(0, lambda: self._update_status(f"Error during collection: {str(e)}\n"))
                self.root.after(0, lambda: self.collect_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.register_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.start_preview_btn.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=collection_thread)
        thread.daemon = True
        thread.start()
    
    def _register_person(self):
        """Register a person in the database"""
        # Check if database manager is initialized
        if self.db_manager is None:
            messagebox.showwarning("Database Error", 
                                  "No database connection. Please connect to the database first.")
            return
            
        registration_number = self.registration_number_var.get().strip()
        full_name = self.full_name_var.get().strip()
        mobile_number = self.mobile_number_var.get().strip()
        
        if not registration_number:
            messagebox.showerror("Error", "Please enter a Registration Number")
            return
        
        if not full_name:
            full_name = registration_number  # Use registration number as name if not provided
        
        # Update status
        self._update_status(f"Registering {registration_number} ({full_name})...\n")
        
        # Register person
        result = self.db_manager.register_person(registration_number, full_name, mobile_number)
        
        if result:
            self._update_status(f"Successfully registered {full_name} in the database.\n")
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
    def _toggle_recognition(self):
        """Toggle between starting and stopping recognition"""
        if self.recognition_active:
            # Stop recognition
            self.stop_recognition = True
            self.start_recog_btn.config(state=tk.DISABLED)
            self._update_recog_status("Stopping recognition...\n")
        else:
            # Start recognition
            self._start_recognition()
    
    def _start_recognition(self):
        """Start face recognition process"""
        try:
            threshold = float(self.threshold_var.get())
            if threshold < 0.1 or threshold > 0.9:
                raise ValueError("Threshold must be between 0.1 and 0.9")
            
            # Get selected camera index
            camera_index = int(self.camera_var.get())
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            return
        
        mark_attendance = self.mark_attendance_var.get()
        
        # Update status
        self._update_recog_status("Starting face recognition...\n")
        self._update_recog_status(f"Camera: {camera_index}, Threshold: {threshold}, Mark Attendance: {mark_attendance}\n")
        
        # Reset stop flag and set active flag
        self.stop_recognition = False
        self.recognition_active = True
        
        # Update button text
        self.start_recog_btn.config(text="Stop Recognition")
        
        # Disable preview button during recognition
        self.start_recog_preview_btn.config(state=tk.DISABLED)
        
        # Stop any ongoing preview
        self._stop_camera()
        
        # Run recognition in a separate thread
        def recognition_thread():
            try:
                recognizer = recognition_system.FaceRecognitionSystem(threshold=threshold)
                recognizer.recognize_from_webcam(
                    camera_index=camera_index,
                    mark_attendance=mark_attendance,
                    preview_widget=self.recog_preview_canvas,
                    convert_func=self._convert_cv_to_tkimage,
                    app_root=self.root,
                    status_callback=self._update_recog_status_thread_safe,
                    stop_flag=lambda: self.stop_recognition  # Pass the stop flag function
                )
                
                # Reset flags and update UI
                self.recognition_active = False
                self.stop_recognition = False
                
                # Re-enable buttons
                self.root.after(0, lambda: self.start_recog_btn.config(state=tk.NORMAL, text="Start Recognition"))
                self.root.after(0, lambda: self.start_recog_preview_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self._update_recog_status("Recognition stopped.\n"))
                
                # Refresh reports tab if attendance was marked
                if mark_attendance:
                    self.root.after(0, lambda: self._load_attendance_report())
            except Exception as e:
                # Handle errors
                self.recognition_active = False
                self.stop_recognition = False
                self.root.after(0, lambda: self._update_recog_status(f"Error during recognition: {str(e)}\n"))
                self.root.after(0, lambda: self.start_recog_btn.config(state=tk.NORMAL, text="Start Recognition"))
                self.root.after(0, lambda: self.start_recog_preview_btn.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=recognition_thread)
        thread.daemon = True
        thread.start()
    
    def _update_recog_status_thread_safe(self, message):
        """Update recognition status text widget from a thread"""
        self.root.after(0, lambda: self._update_recog_status(message))
    
    def _update_recog_status(self, message):
        """Update recognition status text widget"""
        self.recog_status_text.config(state=tk.NORMAL)
        self.recog_status_text.insert(tk.END, message)
        self.recog_status_text.see(tk.END)
        self.recog_status_text.config(state=tk.DISABLED)
    
    # Management tab functions
    def _refresh_person_list(self):
        """Refresh the list of registered persons in the management tab"""
        # Clear the current list
        for item in self.person_tree.get_children():
            self.person_tree.delete(item)
        
        # Check if database manager is initialized
        if self.db_manager is None:
            print("Database manager not initialized. Cannot refresh person list.")
            messagebox.showwarning("Database Error", 
                                  "No database connection. Please connect to the database first.")
            return
            
        try:
            # Debug statement
            print("Refreshing person list...")
            
            # Get the list of registered persons
            persons = self.db_manager.list_registered_persons()
            print(f"Retrieved {len(persons)} persons from database")
            
            # Add them to the tree view
            for reg_num, info in persons.items():
                full_name = info.get("full_name", "")
                mobile = info.get("mobile_number", "")
                sample_count = info.get("sample_count", "0")
                reg_date = info.get("registration_date", "")
                
                print(f"Adding to tree: {reg_num}, {full_name}, {mobile}, {sample_count}, {reg_date}")
                
                self.person_tree.insert("", tk.END, values=(
                    reg_num,
                    full_name,
                    mobile,
                    sample_count,
                    reg_date
                ))
            
            print(f"Added {len(persons)} entries to the tree view")
                    
        except Exception as e:
            print(f"Error in _refresh_person_list: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load registered persons: {str(e)}")
    
    def _remove_selected_person(self):
        """Remove selected person from the database"""
        # Check if database manager is initialized
        if self.db_manager is None:
            messagebox.showwarning("Database Error", 
                                  "No database connection. Please connect to the database first.")
            return
            
        selection = self.person_tree.selection()
        if not selection:
            messagebox.showinfo("Information", "Please select a person to remove")
            return
        
        # Get the selected person's registration number
        registration_number = self.person_tree.item(selection[0], "values")[0]
        
        # Confirm removal
        confirmed = messagebox.askyesno("Confirm", f"Are you sure you want to remove {registration_number}?")
        if not confirmed:
            return
        
        # Remove the person
        result = self.db_manager.remove_person(registration_number)
        
        if result:
            messagebox.showinfo("Success", f"{registration_number} has been removed from the database")
            self._refresh_person_list()
        else:
            messagebox.showerror("Error", f"Failed to remove {registration_number}")
    
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
        """Get all registered persons from the database"""
        persons = set()
        
        try:
            # Use the database manager to get all registered persons
            if self.db_manager:
                person_data = self.db_manager.list_registered_persons()
                for person_id in person_data.keys():
                    persons.add(person_id)
            else:
                print("Warning: Database manager not initialized, cannot get registered persons")
        except Exception as e:
            print(f"Error getting registered persons: {e}")
        
        return persons
    
    # New utility functions for camera preview
    def _convert_cv_to_tkimage(self, cv_image):
        """Convert OpenCV image to Tkinter compatible image"""
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        return ImageTk.PhotoImage(image=pil_image)
    
    def _stop_camera(self):
        """Stop any active camera preview"""
        self.camera_active = False
        if self.preview_after_id:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None
            
        if self.camera_cap and self.camera_cap.isOpened():
            self.camera_cap.release()
            self.camera_cap = None
        
        # Update button text if necessary
        if hasattr(self, 'start_preview_btn'):
            self.start_preview_btn.config(text="Start Preview")
        if hasattr(self, 'start_recog_preview_btn'):
            self.start_recog_preview_btn.config(text="Start Preview")
    
    def _on_tab_change(self, event):
        """Handle tab changes - stop camera when leaving tab"""
        self._stop_camera()
        
        # Also stop recognition if active
        if self.recognition_active:
            self.stop_recognition = True
    
    def _on_closing(self):
        """Handle application closing"""
        self._stop_camera()
        
        # Make sure to stop any running recognition
        self.stop_recognition = True
        
        # Give a moment for threads to clean up
        if self.recognition_active:
            self._update_recog_status("Shutting down recognition system...\n")
            self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()
    
    def _start_reg_preview(self):
        """Start camera preview in registration tab"""
        if self.camera_active:
            self._stop_camera()
            self.start_preview_btn.config(text="Start Preview")
            return
        
        try:
            camera_index = int(self.reg_camera_var.get())
            self.camera_cap = cv2.VideoCapture(camera_index)
            
            if not self.camera_cap.isOpened():
                self._update_status(f"Error: Could not open camera {camera_index}.")
                return
                
            self.camera_active = True
            self._update_reg_preview()
            self.start_preview_btn.config(text="Stop Preview")
            
        except Exception as e:
            self._update_status(f"Error starting camera preview: {str(e)}")
    
    def _update_reg_preview(self):
        """Update registration tab preview with camera frame"""
        if not self.camera_active or not self.camera_cap or not self.camera_cap.isOpened():
            return
            
        ret, frame = self.camera_cap.read()
        if ret:
            # Get canvas dimensions
            canvas_width = self.reg_preview_canvas.winfo_width()
            canvas_height = self.reg_preview_canvas.winfo_height()
            
            # Resize frame to fit canvas if dimensions are valid
            if canvas_width > 1 and canvas_height > 1:
                frame = cv2.resize(frame, (canvas_width, canvas_height))
                
            # Convert frame to Tkinter image
            self.reg_preview_img = self._convert_cv_to_tkimage(frame)
            self.reg_preview_canvas.create_image(0, 0, image=self.reg_preview_img, anchor=tk.NW)
        
        # Continue updating
        self.preview_after_id = self.root.after(33, self._update_reg_preview)  # ~30 FPS
    
    def _start_recog_preview(self):
        """Start camera preview in recognition tab"""
        if self.camera_active:
            self._stop_camera()
            self.start_recog_preview_btn.config(text="Start Preview")
            return
        
        try:
            camera_index = int(self.camera_var.get())
            self.camera_cap = cv2.VideoCapture(camera_index)
            
            if not self.camera_cap.isOpened():
                self._update_recog_status(f"Error: Could not open camera {camera_index}.")
                return
                
            self.camera_active = True
            self._update_recog_preview()
            self.start_recog_preview_btn.config(text="Stop Preview")
            
        except Exception as e:
            self._update_recog_status(f"Error starting camera preview: {str(e)}")
    
    def _update_recog_preview(self):
        """Update recognition tab preview with camera frame"""
        if not self.camera_active or not self.camera_cap or not self.camera_cap.isOpened():
            return
            
        ret, frame = self.camera_cap.read()
        if ret:
            # Get canvas dimensions
            canvas_width = self.recog_preview_canvas.winfo_width()
            canvas_height = self.recog_preview_canvas.winfo_height()
            
            # Resize frame to fit canvas if dimensions are valid
            if canvas_width > 1 and canvas_height > 1:
                frame = cv2.resize(frame, (canvas_width, canvas_height))
                
            # Convert frame to Tkinter image
            self.recog_preview_img = self._convert_cv_to_tkimage(frame)
            self.recog_preview_canvas.create_image(0, 0, image=self.recog_preview_img, anchor=tk.NW)
        
        # Continue updating
        self.preview_after_id = self.root.after(33, self._update_recog_preview)  # ~30 FPS

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemApp(root)
    root.mainloop()