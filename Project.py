import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import customtkinter as ctk
from tkinter import messagebox
import os

# -------------------------------
# 1. CORE LOGIC
# -------------------------------

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Check if recipes.csv exists
if not os.path.exists("recipes.csv"):
    messagebox.showerror(
        "File Error",
        "The 'recipes.csv' file was not found. "
        "Please ensure it is in the same directory."
    )
    exit()

# Load dataset
try:
    df = pd.read_csv(
        "recipes.csv",
        on_bad_lines='skip',
        encoding='utf-8',
        engine='python'
    )
    df['ingredients'] = df['ingredients'].astype(str).str.lower()
except Exception as e:
    messagebox.showerror(
        "Data Error",
        f"Failed to load or process recipes.csv: {e}"
    )
    exit()

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])


def recommend_recipes(user_ingredients, diet_pref="any"):
    """Recommends recipes based on user input using Cosine Similarity."""
    if not user_ingredients.strip():
        return pd.DataFrame()

    try:
        user_vec = vectorizer.transform([user_ingredients.lower()])
    except ValueError:
        return pd.DataFrame()

    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    temp_df = df.copy()
    temp_df['score'] = similarity

    if diet_pref.lower() in ["veg", "nonveg"]:
        results = temp_df[temp_df['diet'].str.lower() == diet_pref.lower()]
    else:
        results = temp_df.copy()

    results = results[results['score'] > 0.05].sort_values(
        by="score", ascending=False
    )
    return results.head(5)


# -------------------------------
# 2. GUI INTERFACE
# -------------------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window dimensions
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        window_w = int(screen_w * 0.75)
        window_h = int(screen_h * 0.75)

        # Center window on screen
        pos_x = (screen_w - window_w) // 2
        pos_y = (screen_h - window_h) // 4

        # Window setup
        self.title("Smart Recipe Recommender 🍲")
        self.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")
        self.resizable(False, False)

        # Layout config
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Title label
        self.title_label = ctk.CTkLabel(
            self,
            text=" Recipe Recommender System",
            font=ctk.CTkFont(family="Arial", size=24, weight="bold"),
            text_color="#3CB371"
        )
        self.title_label.grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="ew"
        )

        # Input Frame
        self.input_frame = ctk.CTkFrame(self, corner_radius=15)
        self.input_frame.grid(
            row=1, column=0, padx=30, pady=10, sticky="ew"
        )
        self.input_frame.grid_columnconfigure((0, 1), weight=1)

        # Ingredients input
        self.ing_label = ctk.CTkLabel(
            self.input_frame,
            text="Enter Available Ingredients (e.g., rice, onion, chicken):",
            font=ctk.CTkFont(weight="bold")
        )
        self.ing_label.grid(
            row=0, column=0, padx=10, pady=(10, 0), sticky="w"
        )

        self.entry_ingredients = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="rice, tomato, onion"
        )
        self.entry_ingredients.grid(
            row=1, column=0, padx=10, pady=(5, 10), sticky="ew"
        )

        # Diet preference dropdown
        self.diet_label = ctk.CTkLabel(
            self.input_frame,
            text="Select Diet Preference:",
            font=ctk.CTkFont(weight="bold")
        )
        self.diet_label.grid(
            row=0, column=1, padx=10, pady=(10, 0), sticky="w"
        )

        self.diet_choice = ctk.CTkComboBox(
            self.input_frame,
            values=["any", "veg", "nonveg"],
            state="readonly"
        )
        self.diet_choice.set("any")
        self.diet_choice.grid(
            row=1, column=1, padx=10, pady=(5, 10), sticky="ew"
        )

        # Recommend button
        self.btn_recommend = ctk.CTkButton(
            self.input_frame,
            text="🔍 Get Recipes",
            command=self.show_recommendations,
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            fg_color="#3CB371",
            hover_color="#2E8B57"
        )
        self.btn_recommend.grid(
            row=2, column=0, columnspan=2, padx=10, pady=(5, 15), sticky="ew"
        )

        # Output Frame
        self.output_frame = ctk.CTkFrame(self, corner_radius=15)
        self.output_frame.grid(
            row=2, column=0, padx=30, pady=(10, 20), sticky="nsew"
        )
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_rowconfigure(1, weight=1)

        # Output title
        self.output_title = ctk.CTkLabel(
            self.output_frame,
            text="Top Recipe Suggestions",
            font=ctk.CTkFont(family="Arial", size=16, weight="bold")
        )
        self.output_title.grid(
            row=0, column=0, padx=10, pady=(10, 5), sticky="w"
        )

        # Output textbox
        self.text_output = ctk.CTkTextbox(
            self.output_frame,
            wrap="word",
            font=("Courier New", 11),
            activate_scrollbars=True,
            corner_radius=10
        )
        self.text_output.grid(
            row=1, column=0, padx=10, pady=(5, 10), sticky="nsew"
        )

    def show_recommendations(self):
        """Handles button click and displays top recipe suggestions."""
        ingredients = self.entry_ingredients.get()
        diet_pref = self.diet_choice.get()

        if not ingredients.strip():
            messagebox.showerror(
                "Input Error",
                "Please enter some ingredients "
                "(e.g., rice, tomato, onion)!"
            )
            return

        self.btn_recommend.configure(state="disabled", text="Processing...")
        self.text_output.delete("0.0", "end")

        results = recommend_recipes(ingredients, diet_pref)

        if results.empty or results['score'].max() < 0.1:
            self.text_output.insert(
                "0.0",
                "⚠️ No strong matches found. "
                "Try modifying your ingredients or preferences."
            )
        else:
            output_text = (
                "✨ Top Recipe Suggestions (Ranked by Match Score):\n\n"
            )
            for i, (_, row) in enumerate(results.iterrows(), start=1):
                output_text += (
                    f"⭐ RANK {i}: {row['name']} (Score: {row['score']:.2f})\n"
                    f"   - Ingredients: {row['ingredients']}\n"
                    f"   - Time: {row['time']} mins\n"
                    f"   - Calories: {row['calories']}\n"
                    f"   - Steps: {row['steps']}\n"
                    + "-" * 50 + "\n"
                )

            self.text_output.insert("0.0", output_text)

        self.btn_recommend.configure(state="normal", text="🔍 Get Recipes")


# -------------------------------
# 3. APP RUNNER
# -------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
