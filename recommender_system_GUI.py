import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import requests
import recommender_backend as rb

# Create the main window
root = tk.Tk()
root.title("Movie Recommender System")
root.geometry("800x800")

# Center-align the title and increase the font size
title_label = ttk.Label(root, text="Top-N Recommender System", font=("Helvetica", 20), anchor="center")
title_label.grid(column=0, row=0, pady=20, columnspan=2)  # Use columnspan to span across both columns

# User ID input
ttk.Label(root, text="Enter User ID:").grid(column=0, row=1, padx=10, pady=10)
user_id = ttk.Entry(root)
user_id.grid(column=1, row=1, padx=10, pady=10)

# Recommender system choice
ttk.Label(root, text="Choose Recommender:").grid(column=0, row=2, padx=10, pady=10)
recommender_choice = ttk.Combobox(root, values=["Genre Based", "Weighted Genre Based","Year Based", "Weighted Year Based",
                                                "Combined Content Based", "User Based", "Item Based", "SVD", "NCF"], state="readonly")
recommender_choice.grid(column=1, row=2, padx=10, pady=10)
recommender_choice.set("Genre Based")  # Default choice

# Scrollable frame for movie posters
scrollable_frame = ttk.Frame(root)
scrollable_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

canvas = tk.Canvas(scrollable_frame, width=750, height=300)
canvas.grid(row=0, column=0)
scrollbar_h = ttk.Scrollbar(scrollable_frame, orient="horizontal", command=canvas.xview)  # Horizontal scrollbar
scrollbar_h.grid(row=1, column=0, sticky="ew")
canvas.configure(xscrollcommand=scrollbar_h.set)
inner_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=inner_frame, anchor="nw")

# Hover text to display movie description
hover_text = tk.StringVar()
hover_label = ttk.Label(root, textvariable=hover_text, wraplength=700)
hover_label.grid(row=5, column=0, columnspan=2, pady=10)


# Display movie recommendations
def display_movies(recommendations):
    global img_tk
    for widget in inner_frame.winfo_children():
        widget.destroy()
    for index, movie_data in enumerate(recommendations):
        image_url = movie_data.get('poster_url', None)
        if image_url:
            try:
                response = requests.get(image_url)
                response.raise_for_status()  # Raise an error for bad responses
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((100, 150))
                img_tk = ImageTk.PhotoImage(img)
                img_label = ttk.Label(inner_frame, image=img_tk)
                img_label.image = img_tk
                img_label.grid(row=0, column=index, padx=5, pady=5)
            except requests.RequestException as e:
                print(f"Failed to fetch poster for {movie_data['title']}. Error: {e}")

        img_label = ttk.Label(inner_frame, image=img_tk, width=150)  # Adjusted size
        img_label.image = img_tk
        img_label.grid(row=0, column=index, padx=5, pady=5)
        title_label = ttk.Label(inner_frame, text=movie_data['title'], wraplength=150, anchor="center")
        title_label.grid(row=1, column=index, padx=5)
        year_label = ttk.Label(inner_frame, text=f"({movie_data['year']})", anchor="center")
        year_label.grid(row=2, column=index, padx=5)

        # Bind hover event to show description
        for widget in [img_label, title_label, year_label]:
            widget.bind("<Enter>", lambda event, text=movie_data['description']: hover_text.set(text))
            widget.bind("<Leave>", lambda event: hover_text.set(""))


# Generate recommendations
def generate_recommendations():
    global recommendations
    uid = int(user_id.get())
    recommender = recommender_choice.get()
    if recommender == "Genre Based":
        recommendations = rb.get_movie_recommendations(uid, rb.genre_based_scores)
    elif recommender == "Weighted Genre Based":
        recommendations = rb.get_movie_recommendations(uid, rb.weighted_genre_based_scores)
    elif recommender == "Year Based":
        recommendations = rb.get_movie_recommendations(uid, rb.year_based_scores)
    elif recommender == "Weighted Year Based":
        recommendations = rb.get_movie_recommendations(uid, rb.weighted_year_based_scores)
    elif recommender == "Combined Content Based":
        recommendations = rb.get_movie_recommendations(uid, lambda user: rb.combined_content_based_scores(user))
    elif recommender == "User Based":
        recommendations = rb.get_movie_recommendations(uid, rb.user_user_predicted_scores_optimized)
    elif recommender == "Item Based":
        recommendations = rb.get_movie_recommendations(uid, rb.item_based_predicted_scores)
    elif recommender == "SVD":
        recommendations = rb.get_movie_recommendations(uid, rb.svd_movie_scores)
    else:
        recommendations = rb.get_ncf_movie_recommendations(uid)
    display_movies(recommendations)


recommend_button = ttk.Button(root, text="Generate Recommendations", command=generate_recommendations)
recommend_button.grid(column=0, row=3, columnspan=2, pady=20)

root.mainloop()
