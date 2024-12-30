import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil

def upload_file():
    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("All Files", "*.*"), ("Text Files", "*.txt"), ("Images", "*.png;*.jpg;*.jpeg")]
    )
    if file_path:  # Check if a file is selected
        # Ensure the 'uploads' folder exists
        upload_folder = os.path.join(os.getcwd(), "ImagesAttendance")
        os.makedirs(upload_folder, exist_ok=True)

        # Get the file name and target path
        file_name = os.path.basename(file_path)
        target_path = os.path.join(upload_folder, file_name)

        try:
            # Copy the file to the 'uploads' folder
            shutil.copy(file_path, target_path)
            messagebox.showinfo("File Uploaded", f"File saved to:\n{target_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
    else:
        messagebox.showwarning("No File Selected", "Please select a file to upload.")

# Create the main window
root = tk.Tk()
root.title("File Upload")

# Create an upload button
upload_button = tk.Button(root, text="Upload File", command=upload_file, font=("Arial", 14))
upload_button.pack(pady=20)

# Run the application
root.geometry("300x150")
root.mainloop()
