import tkinter as tk
from tkinter import ttk
import csv
import os
import datetime

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System Dashboard")
        self.root.geometry("800x600")
        
        # Create UI elements
        self.create_widgets()
        
        # Load today's attendance by default
        self.load_attendance()
    
    def create_widgets(self):
        # Date selection
        date_frame = ttk.Frame(self.root, padding="10")
        date_frame.pack(fill=tk.X)
        
        ttk.Label(date_frame, text="Select Date:").pack(side=tk.LEFT)
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_var = tk.StringVar(value=today)
        date_entry = ttk.Entry(date_frame, textvariable=self.date_var, width=15)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        load_btn = ttk.Button(date_frame, text="Load", command=self.load_attendance)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Attendance table
        table_frame = ttk.Frame(self.root, padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview
        self.tree = ttk.Treeview(table_frame, columns=("ID", "Time", "Status"), show="headings")
        self.tree.heading("ID", text="Registration Number")
        self.tree.heading("Time", text="Check-in Time")
        self.tree.heading("Status", text="Status")
        
        self.tree.column("ID", width=100)
        self.tree.column("Time", width=100)
        self.tree.column("Status", width=100)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Stats
        stats_frame = ttk.Frame(self.root, padding="10")
        stats_frame.pack(fill=tk.X)
        
        self.total_var = tk.StringVar(value="Total: 0")
        self.present_var = tk.StringVar(value="Present: 0")
        self.absent_var = tk.StringVar(value="Absent: 0")
        
        ttk.Label(stats_frame, textvariable=self.total_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(stats_frame, textvariable=self.present_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(stats_frame, textvariable=self.absent_var).pack(side=tk.LEFT, padx=20)
    
    def load_attendance(self):
        # Clear current data
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        date = self.date_var.get()
        file_path = f'attendance_{date}.csv'
        
        if not os.path.exists(file_path):
            # No attendance for this date
            self.total_var.set("Total: 0")
            self.present_var.set("Present: 0")
            self.absent_var.set("Absent: 0")
            return
        
        # Load attendance data
        present_count = 0
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 3:
                    self.tree.insert("", tk.END, values=row)
                    if row[2] == "Present":
                        present_count += 1
        
        # Update stats
        total_students = len(self.get_all_registered_students())
        self.total_var.set(f"Total: {total_students}")
        self.present_var.set(f"Present: {present_count}")
        self.absent_var.set(f"Absent: {total_students - present_count}")
    
    def get_all_registered_students(self):
        # This would fetch from your student database
        # For demo, just read from metadata file
        students = set()
        if os.path.exists('face_db/metadata.txt'):
            with open('face_db/metadata.txt', 'r') as f:
                for line in f:
                    registration_number = line.strip().split(',')[0]
                    students.add(registration_number)
        return students

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()