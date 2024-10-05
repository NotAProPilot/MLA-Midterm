import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# Create a function to load and display the .npz file
def load_npz_file():
    # Open a file dialog to select the .npz file
    file_path = filedialog.askopenfilename(
        title="Select .npz File",
        filetypes=[("NumPy NPZ Files", "*.npz")]
    )
    
    if file_path:
        try:
            # Load the .npz file
            npz_file = np.load(file_path)
            
            # Display the content in the text widget
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"Contents of the .npz file: {file_path}\n\n")
            
            for key in npz_file.files:
                text_widget.insert(tk.END, f"{key}:\n")
                text_widget.insert(tk.END, f"{npz_file[key]}\n\n")
            
            npz_file.close()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    else:
        messagebox.showinfo("No File", "No file selected.")

# Initialize the GUI window
app = tk.Tk()
app.title("NPZ File Loader")
app.geometry("600x400")

# Add a button to open the file dialog
load_button = tk.Button(app, text="Load .npz File", command=load_npz_file)
load_button.pack(pady=20)

# Add a text widget to display the contents of the file
text_widget = tk.Text(app, wrap=tk.NONE)
text_widget.pack(expand=True, fill='both', padx=10, pady=10)

# Add scrollbars to the text widget
scrollbar_y = tk.Scrollbar(app, orient=tk.VERTICAL, command=text_widget.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
text_widget.config(yscrollcommand=scrollbar_y.set)

scrollbar_x = tk.Scrollbar(app, orient=tk.HORIZONTAL, command=text_widget.xview)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
text_widget.config(xscrollcommand=scrollbar_x.set)

# Run the app
app.mainloop()
