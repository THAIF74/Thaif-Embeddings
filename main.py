from openai import OpenAI
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import threading

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

MODEL = "nvidia/nemotron-3-embed-1b"


def get_embedding(text):
    response = client.embeddings.create(
        model=MODEL,
        input=text,
        encoding_format="float"
    )

    return response.data[0].embedding


def calculate_similarity(vector1, vector2):
    a = np.array(vector1)
    b = np.array(vector2)

    score = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return score


def compare_sentences():
    text1 = sentence1_entry.get().strip()
    text2 = sentence2_entry.get().strip()

    if not text1 or not text2:
        messagebox.showwarning(
            "Missing Text",
            "Please enter both sentences."
        )
        return

    compare_button.config(state=tk.DISABLED)
    result_label.config(text="Generating embeddings...")
    score_label.config(text="")

    threading.Thread(
        target=process_similarity,
        args=(text1, text2),
        daemon=True
    ).start()


def process_similarity(text1, text2):
    try:
        vector1 = get_embedding(text1)
        vector2 = get_embedding(text2)

        score = calculate_similarity(vector1, vector2)

        root.after(
            0,
            show_result,
            score
        )

    except Exception as e:
        root.after(
            0,
            show_error,
            str(e)
        )


def show_result(score):
    percentage = score * 100

    if score >= 0.70:
        interpretation = "Highly Similar 🟢"
    elif score >= 0.50:
        interpretation = "Somewhat Similar 🟡"
    else:
        interpretation = "Not Very Similar 🔴"

    result_label.config(
        text=interpretation
    )

    score_label.config(
        text=f"Similarity Score: {score:.4f}\n"
             f"Similarity: {percentage:.2f}%"
    )

    compare_button.config(state=tk.NORMAL)


def show_error(error):
    result_label.config(
        text="Error occurred ❌"
    )

    score_label.config(
        text=error
    )

    compare_button.config(state=tk.NORMAL)


def clear_fields():
    sentence1_entry.delete(0, tk.END)
    sentence2_entry.delete(0, tk.END)

    result_label.config(
        text="Enter two sentences and compare them."
    )

    score_label.config(text="")


# -----------------------------
# GUI
# -----------------------------

root = tk.Tk()

root.title("Thaif Semantic Search")
root.geometry("700x550")

title = tk.Label(
    root,
    text="🧠 Thaif Semantic Search",
    font=("Arial", 22, "bold")
)

title.pack(pady=20)


subtitle = tk.Label(
    root,
    text="Compare the meaning of two sentences using NVIDIA Embeddings",
    font=("Arial", 11)
)

subtitle.pack(pady=5)


# Sentence 1

label1 = tk.Label(
    root,
    text="Sentence 1",
    font=("Arial", 12, "bold")
)

label1.pack(anchor="w", padx=40, pady=(20, 5))


sentence1_entry = tk.Entry(
    root,
    font=("Arial", 12)
)

sentence1_entry.pack(
    fill="x",
    padx=40,
    ipady=8
)


# Sentence 2

label2 = tk.Label(
    root,
    text="Sentence 2",
    font=("Arial", 12, "bold")
)

label2.pack(anchor="w", padx=40, pady=(20, 5))


sentence2_entry = tk.Entry(
    root,
    font=("Arial", 12)
)

sentence2_entry.pack(
    fill="x",
    padx=40,
    ipady=8
)


# Buttons

button_frame = tk.Frame(root)

button_frame.pack(pady=25)


compare_button = tk.Button(
    button_frame,
    text="Compare",
    command=compare_sentences,
    font=("Arial", 12, "bold"),
    padx=25,
    pady=8
)

compare_button.pack(
    side=tk.LEFT,
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    font=("Arial", 12),
    padx=25,
    pady=8
)

clear_button.pack(
    side=tk.LEFT,
    padx=10
)


# Result

result_label = tk.Label(
    root,
    text="Enter two sentences and compare them.",
    font=("Arial", 14, "bold")
)

result_label.pack(pady=10)


score_label = tk.Label(
    root,
    text="",
    font=("Arial", 13)
)

score_label.pack(pady=10)


# Enter key support

sentence2_entry.bind(
    "<Return>",
    lambda event: compare_sentences()
)


root.mainloop()