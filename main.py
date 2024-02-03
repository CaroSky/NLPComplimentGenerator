import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
from MarkovChain import MarkovChain
from preprocessing import ComplimentPreprocessor

# GUI App
class MarkovApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.markov = self.train_model()

        self.title("Compliment generator")
        self.geometry("700x700")

        # Set weights for the main window
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Main centered frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=1, column=1, sticky="nsew")

        self.headline = ttk.Label(self, text="Compliment Generator", font=("Arial", 24))
        self.headline.grid(row=0, column=0, pady=10, padx=10)

        # A control frame
        self.control_frame = ttk.Frame(self)
        self.control_frame.grid(row=1, column=0, pady=10, padx=10)

        # Add widgets to control frame
        self.sentence_count_label = ttk.Label(self.control_frame, text="Number of Sentences:")
        self.sentence_count_label.grid(row=0, column=0, padx=5)

        self.sentence_count_entry = ttk.Entry(self.control_frame)
        self.sentence_count_entry.grid(row=0, column=1, padx=5)

        self.generate_button = ttk.Button(self.control_frame, text="Generate", command=self.generate_sentences)
        self.generate_button.grid(row=0, column=2, padx=5)

        # Add text area to the main window
        self.text_area = tk.Text(self, height=20, width=80)
        self.text_area.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky=tk.W + tk.E)

        # Save and delete buttons
        self.save_button = ttk.Button(self, text="Save", command=self.save_sentences)
        self.save_button.grid(row=3, column=0, columnspan=2, pady=5, padx=10)

        self.delete_button = ttk.Button(self, text="Delete", command=self.delete_sentences)
        self.delete_button.grid(row=4, column=0, columnspan=2, pady=5, padx=10)

        # If anyone wants to load their own dataset
        message = ("Want to generate sentences from your own dataset?\n"
                   "Please write in the header name of the column you want to train the model on\n"
                   "and choose a folder with your files:")
        self.instruction_frame = ttk.Frame(self)
        self.instruction_frame.grid(row=5, column=0, pady=10, padx=10, sticky=tk.W + tk.E)

        self.instruction_label = ttk.Label(self.instruction_frame, text=message)
        self.instruction_label.grid(row=0, column=0, pady=10, padx=10, sticky=tk.W, columnspan=3)

        self.column_header_label = ttk.Label(self.instruction_frame, text="Column Header:")
        self.column_header_label.grid(row=1, column=0, padx=5, sticky=tk.W)

        self.column_header_entry = ttk.Entry(self.instruction_frame)
        self.column_header_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W + tk.E)
        self.column_header_entry.insert(0, "Compliment")  # default value

        self.load_data_button = ttk.Button(self.instruction_frame, text="Load Data", command=self.load_data)
        self.load_data_button.grid(row=1, column=2, pady=5, padx=5, sticky=tk.W)

    def train_model(self, path=None):
        if not path:
            # Use the default data source
            df = pd.read_csv("processed_compliment.csv")
            sentences = df["Compliment"].tolist()
        else:
            column_name = self.column_header_entry.get()  # Get the column name from the entry field only if custom path is given
            sentences = []
            files_processed = []  # List to keep track of the files processed

            for filename in os.listdir(path):
                if filename.endswith(".txt"):
                    with open(os.path.join(path, filename), "r") as file:
                        sentences.extend(file.readlines())
                    files_processed.append(filename)  # Append the filename to list
                elif filename.endswith(".csv"):
                    # If it's a CSV, use pandas to load it
                    df = pd.read_csv(os.path.join(path, filename))
                    if column_name in df.columns:
                        sentences.extend(df[column_name].tolist())
                        files_processed.append(filename)  # Append the filename to list
                    else:
                        # Handle the error where the specified column doesn't exist
                        messagebox.showerror("Error", f"Column '{column_name}' not found in the file {filename}.")
                        return None

        return MarkovChain(sentences)

    def load_data(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            preprocess_option = messagebox.askyesno("Preprocess Data",
                                                    "Would you like to preprocess the data before training?")
            if preprocess_option:
                # Preprocess data
                sentences = self.preprocess_data(folder_selected)
            else:
                sentences = []
                for filename in os.listdir(folder_selected):
                    if filename.endswith(".txt"):
                        with open(os.path.join(folder_selected, filename), "r") as file:
                            sentences.extend(file.readlines())
                    elif filename.endswith(".csv"):
                        column_name = self.column_header_entry.get()
                        df = pd.read_csv(os.path.join(folder_selected, filename))
                        if column_name in df.columns:
                            sentences.extend(df[column_name].tolist())
                        else:
                            messagebox.showerror("Error", f"Column '{column_name}' not found in the file {filename}.")

            # Train the model
            self.markov = MarkovChain(sentences)

    def preprocess_data(self, folder_path):
        # This function preprocesses the data and returns the list of sentences
        all_sentences = []
        column_name = self.column_header_entry.get()  # Get the column name from the entry field

        for filename in os.listdir(folder_path):
            if filename.endswith((".txt", ".csv")):
                filepath = os.path.join(folder_path, filename)
                preprocessor = ComplimentPreprocessor(filepath, column_name=column_name)
                preprocessor.preprocess()

                # Directly access the preprocessed data and extend all_sentences
                processed_sentences = preprocessor.data[column_name].astype(str).tolist()
                all_sentences.extend(processed_sentences)

        return all_sentences

    def generate_sentences(self):
        try:
            count = int(self.sentence_count_entry.get())
            sentences = []
            for i in range(count):
                compliment = self.markov.generate()
                # Capitalize the first letter and add a '.' at the end
                compliment = compliment[0].upper() + compliment[1:] + "."
                # Precede the compliment with a number and add it to the list
                sentences.append(f"- {compliment}")
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, '\n'.join(sentences))
        except ValueError:
            # Handle cases where the user didn't input a valid number
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, "Please enter a valid number for the number of sentences.")

    def save_sentences(self):
        with open("generated_compliments.txt", "a") as file:
            file.write(self.text_area.get(1.0, tk.END))
        messagebox.showinfo("Success", "Content saved successfully!")

    def delete_sentences(self):
        self.text_area.delete(1.0, tk.END)


if __name__ == "__main__":
    app = MarkovApp()
    app.mainloop()

