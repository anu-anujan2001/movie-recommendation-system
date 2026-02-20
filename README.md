# 🎬 Movie Recommendation System (Streamlit + ML)

A **Content-Based Movie Recommendation System** built using **Python, Streamlit, and Scikit-learn**.  
It recommends movies similar to a user’s favorite movie using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🚀 Features

- ✅ Content-based movie recommendations
- ✅ Handles spelling mistakes using `difflib.get_close_matches()`
- ✅ Suggests Top 30 similar movies
- ✅ Simple and clean Streamlit UI
- ✅ Works with common movie datasets (movies.csv)

---

## 🧠 How It Works

1. Select important text features:
   - `genres`, `keywords`, `tagline`, `cast`, `director`
2. Combine all features into a single text column
3. Convert text into numeric vectors using **TF-IDF**
4. Calculate similarity between movies using **Cosine Similarity**
5. Recommend the most similar movies to the user’s input

---

## 🛠️ Technologies Used

- Python
- Pandas
- Streamlit
- Scikit-learn
- Difflib

---

## 📁 Project Structure

```bash
movie-recommendation-system/
│
├── app.py
├── movies.csv
├── requirements.txt
├── .gitignore
└── README.md
