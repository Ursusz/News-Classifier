# News Classifier - Online Disinformation Detection

## About The Project

**News Classifier** is a Flask/Python web application designed to help users identify fake news (disinformation) from real news. The app analyzes text or URLs using two Machine Learning models: a **Random Forest Classifier (RFC)** and **BERT**. It includes user authentication, prediction history, and smart caching.

## Key Features

* **News Classification:** Determines if an article is "FAKE" or "REAL."
* **Diverse ML Models:** Utilizes RFC or BERT for predictions.
* **Authentication:** User login/registration system.
* **History:** Saves user prediction history.
* **Caching:** Optimized for recurrent URL analysis.

## Technologies Used

* **Backend:** Python 3, Flask, SQLAlchemy, NLTK, Pandas, PyTorch, Hugging Face Transformers, scikit-learn.
* **Database:** PostgreSQL.
* **Web Scraping:** `requests`, BeautifulSoup4.
* **Frontend:** HTML, CSS (SCSS), JavaScript, Bootstrap 5.

## How To Run Locally

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/YourUsername/News-Classifier.git](https://github.com/YourUsername/News-Classifier.git)
    cd News-Classifier
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # or `.\venv\Scripts\activate` on Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Database:**
    * Ensure PostgreSQL is running and create a database.
    * Create a `.env` file with `DATABASE_URL` and `SECRET_KEY`.
5.  **Initialize Database:**
    ```bash
    python -c "from app import create_app, db; app = create_app(); with app.app_context(): db.create_all()"
    ```
6.  **Run Application:**
    ```bash
    python run.py
    ```
    The application will be accessible at `http://127.0.0.1:8080/`.
    *(Pre-trained ML models are included in `app/nlps/`.)*

## Contributions

Contributions are welcome! Feel free to open an Issue or a Pull Request.
