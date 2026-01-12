# 📰 AI News Assistant & Press Review

**AI News Assistant** is a comprehensive Python tool designed to aggregate real-time news and perform advanced analysis using Natural Language Processing (NLP).

Unlike standard news aggregators, this assistant reads the articles for you. It leverages Hugging Face Transformers to generate summaries, detect sentiment, extract entities (NER), and visualize daily trending topics.

## ✨ Key Features

* **🌐 Smart Aggregation:** Fetches top headlines based on categories (Tech, Business, Science, etc.) using NewsAPI.
* **🧠 AI-Powered Analysis:**
    * **Summarization:** Condenses long articles into concise briefings using the `DistilBART` model.
    * **Sentiment Analysis:** Evaluates the tone (Positive/Negative) of specific articles and the general daily mood.
    * **NER (Named Entity Recognition):** Identifies People, Organizations, and Locations mentioned in the text.
* **☁️ Visualizations:** Generates a **Word Cloud** to visualize the most frequent topics of the day.
* **📥 Smart Scraping:** Uses `trafilatura` to extract clean content from web pages, bypassing ads and cluttered layouts.
* **🛡️ Paywall Filter:** Automatically skips domains known for strict paywalls (e.g., WSJ, Bloomberg) to ensure smooth analysis.

## 🛠️ Technology Stack

* **Core:** Python 3.8+
* **NLP & AI:** Hugging Face Transformers, PyTorch/TensorFlow, NLTK.
* **Data Source:** NewsAPI, the Web.
* **Utilities:** Trafilatura (Scraping), WordCloud & Matplotlib (Data Viz).

## 🚀 Installation & Setup

To run this application, ensure you have a Python environment set up on your machine.

1.  **Install Dependencies:**
    Download the source code and install the required Python libraries listed in the `requirements.txt` file included in this repository.

2.  **Model Download:**
    The first time you launch the application, it will automatically download the necessary AI models (Sentiment, Summarization, NER) and NLTK data. This requires an active internet connection and may take a few minutes depending on your speed.

## ⚙️ Configuration

You must provide a valid API Key from **NewsAPI** for the application to function.

1.  Obtain a free API Key at [https://newsapi.org](https://newsapi.org).
2.  Open the `main.py` file in your code editor.
3.  Locate the `main()` function and assign your key to the `API_KEY` variable:

```python
def main():
    API_KEY = 'YOUR_ACTUAL_API_KEY_HERE' 
    # ...
```

## 🖥️ Usage

Run the main script to launch the interactive interface.

**Workflow:**
1.  **Select a Category:** Choose your area of interest (e.g., General, Technology, Health) from the numbered menu.
2.  **Daily Briefing:** The assistant will first download the latest headlines. It will then display a **Word Cloud** window showing trending topics and print a global sentiment score.
3.  **Article Selection:** Close the Word Cloud window to view the list of headlines.
4.  **Deep Analysis:** Enter the number of a specific article to trigger the full AI pipeline. The system will scrape the full text, generate a summary, and extract key entities (People, Organizations, Locations).

## ⚠️ System Requirements & Performance

* **Memory (RAM):** At least **4GB** is recommended to handle the Transformer models (BERT and BART) efficiently.
* **Hardware Acceleration:** The application is built to automatically detect **NVIDIA GPUs** (via CUDA). If available, analysis will be significantly faster. If not, it defaults to CPU processing.
* **Internet:** An active connection is required to fetch metadata from NewsAPI, download models (on first run), and scrape article content.

## 📄 License

This project is distributed under the **MIT License**.




