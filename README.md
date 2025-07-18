# 🎬 Movie Recommender System (Streamlit App)

This is a **Movie Recommendation System** built using **Python**, **Streamlit**, and **scikit-learn**.  
The app takes a `User ID`, identifies their favorite movie (based on highest rating), and recommends similar movies using a **Collaborative Filtering** approach with **k-Nearest Neighbors**.

---

## 🚀 Features

- 🔢 Enter a `User ID` to receive personalized movie recommendations  
- 🎚️ Choose how many movies to recommend using a slider  
- 🎭 Optional genre filter (e.g. Action, Comedy, Drama)  
- ⚡ Fast and efficient with sparse matrix representation  
- 🌐 Simple and modern browser-based UI with **Streamlit**

---

## 📂 Dataset

This app uses the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/) containing:
- `ratings.csv` – userId, movieId, rating, timestamp  
- `movies.csv` – movieId, title, genres  

Place these files in the **same directory** as the app script.

---

## 📦 Installation

Install required Python packages:

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## 🧠 How It Works

1. The user enters their `userId`  
2. The app identifies their **top-rated movie**  
3. Using **cosine similarity** and **k-NN** on the user-item matrix, it finds similar movies  
4. Optionally filters by **genre**  
5. Displays recommendations in the browser  

---

## ▶️ How to Run

Run the app with:

```bash
streamlit run movie_recommender_app.py
```

Then open the app in your browser at:

```
http://localhost:8501
```

---

## ✨ Example

**Input:**  
User ID: `128`  
Top-rated movie: *Star Wars (1977)*

**Recommendations:**
- The Empire Strikes Back (1980)  
- Return of the Jedi (1983)  
- Raiders of the Lost Ark (1981)  
- ...

---

## 🛠️ To-Do

- [ ] Use multiple top-rated movies for recommendations  
- [ ] Show movie posters and release years  
- [ ] Add user login / favorite system  
- [ ] Deploy online via Streamlit Cloud or Hugging Face Spaces  

---

## 📄 License

This project is for **educational and research use only**.

---

## ❤️ Built With

- [Streamlit](https://streamlit.io/)  
- [scikit-learn](https://scikit-learn.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)


