import sys
import requests
import trafilatura
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from newsapi import NewsApiClient
from newspaper import Article
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

class NewsAssistant:
    def __init__(self, api_key):
        print("--- Initialising NewsAssistant ---")
        self.newsapi = NewsApiClient(api_key=api_key)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        print("Loading AI models...")
        # 1. Sentiment Analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # 2. Summarization
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # 3. Named Entity Recognition
        print("Loading NER model...")
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        
        print("All models ready!\n")

    def get_top_headlines(self, category='general', page_size=20):
        try:
            top_headlines = self.newsapi.get_top_headlines(
                category=category, language='en', page_size=page_size
            )
            return top_headlines['articles']
        except Exception as e:
            print(f"API error: {e}")
            return []

    def preprocess_text(self, text):
        """Preprocessing Steps - Cleaning for Keywords e WordCloud"""
        if not text: return []
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        custom_stops = {'news', 'report', 'says', 'new', 'update', 'live', 'said', 'us', 'could', 'would'}
        stop_words.update(custom_stops)
        filtered = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
        return filtered

    def generate_daily_briefing(self, category='general'):
        """Create the 'Press Review' with Word Cloud and Daily Sentiment."""
        print(f"\n--- 📊 GENERATING DAILY PRESS REVIEW ({category.upper()}) ---")
        print("Downloading and analysing news...")
        
        articles = self.get_top_headlines(category=category, page_size=30)
        
        if not articles:
            print("News not found.")
            return []

        all_text = ""
        sentiments = []
        
        for art in articles:
            content = f"{art['title']} {art['description'] or ''}" 
            all_text += content + " "
            try:
                res = self.sentiment_analyzer(art['title'][:512])[0]
                sentiments.append(res['label'])
            except:
                pass

        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        if total > 0:
            pos_pct = (sentiment_counts['POSITIVE'] / total) * 100
            neg_pct = (sentiment_counts['NEGATIVE'] / total) * 100
            sentiment_summary = f"Global Sentiment: {pos_pct:.1f}% POSITIVE | {neg_pct:.1f}% NEGATIVE"
        else:
            sentiment_summary = "Impossible to calculate sentiment "

        print("Generating Word Cloud...")
        clean_tokens = self.preprocess_text(all_text)
        clean_text_string = " ".join(clean_tokens)
        
        if not clean_text_string:
            print("Not enough text to generate a Word Cloud.")
            return articles

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_text_string)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"TOPICS OF THE DAY - {category.upper()}\n{sentiment_summary}", fontsize=14, color='darkblue')
        
        print(f"\n>>> {sentiment_summary}")
        print(">>> A Word Cloud window is open. Close to continue.")
        plt.show()
        
        return articles

    def analyze_single_article(self, url):
        """Summarization + Sentiment + Keywords + NER"""
        try:
            print(f"   Trying smart download from: {url[:40]}...")
            
            # 1. ACQUISITION
            downloaded = trafilatura.fetch_url(url)
            
            if downloaded:
                full_text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            else:
                # Fallback: if trafilatura fails, try with requests
                try:
                    response = requests.get(url, headers=self.headers, timeout=20)
                    full_text = trafilatura.extract(response.text)
                except:
                    full_text = None
                    
            if not full_text or len(full_text) < 100: 
                return None

            # 2. SUMMARIZATION
            input_len = len(full_text.split())
            max_len = min(130, input_len // 2)
            summary = self.summarizer(full_text, max_length=max_len, min_length=30, truncation=True)[0]['summary_text']
            
            # 3. SENTIMENT ANALYSIS
            sentiment = self.sentiment_analyzer(full_text[:512])[0]
            
            # 4. KEYWORDS EXTRACTION
            clean_tokens = self.preprocess_text(full_text)
            keywords = Counter(clean_tokens).most_common(5)

            # 5. NAMED ENTITY RECOGNITION (NER)
            ner_results = self.ner_pipeline(full_text[:1000])
            
            entities = {"PER": set(), "ORG": set(), "LOC": set()}
            
            for entity in ner_results:
                if entity['score'] > 0.80:
                    entity_type = entity['entity_group']
                    if entity_type in entities:
                        entities[entity_type].add(entity['word'])
            
            return {
                "summary": summary,
                "full_text": full_text,
                "sentiment": sentiment,
                "keywords": keywords,
                "entities": entities
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            return None

def main():
    API_KEY = 'f2473c71c92945a0add2e545acdd5ee0' 
    assistant = NewsAssistant(API_KEY)

    while True:
        print("\n" + "="*50)
            print(" 📰 AI NEWS ASSISTANT & PRESS REVIEW")
        print("="*50)
        categories = ['general', 'technology', 'business', 'science', 'health']
        for i, cat in enumerate(categories):
            print(f"{i+1}. {cat.capitalize()}")
        print("q. Esci")
        
        choice = input("\nSelect your area of interest: ")
        if choice.lower() == 'q': break
        
        try:
            cat = categories[int(choice)-1]
        except:
            continue
            
        articles = assistant.generate_daily_briefing(category=cat)
        if not articles: continue

        print("\n--- ARTICLE DETAILS ---")
        for i, art in enumerate(articles[:10]):
            print(f"{i+1}. {art['title']}")
        
        sel = input("\nDo you want to analyse a specific article? (Insert a number o 'b' to go back): ")
        if sel == 'b': continue
        
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(articles):
                selected_article = articles[idx]
                print(f"\n--- In-depth Analysis: {selected_article['title']} ---")
                print("Executing Text Mining pipeline (NER, Sentiment, Summarization)...")
                
                result = assistant.analyze_single_article(selected_article['url'])
                
                if result:
                    print("\n" + "█"*20 + " REPORT " + "█"*20)
                    
                    print(f"\n📝 SUMMARY:\n{result['summary']}")
                    
                    label = result['sentiment']['label']
                    score = result['sentiment']['score']
                    icon = "🟢" if label == 'POSITIVE' else "🔴"
                    print(f"\n{icon} SENTIMENT:\n{label} (Confidence: {score:.4f})")
                    
                    # --- SEZIONE NER ---
                    print("\n🔍 NER ANALYSIS:")
                    
                    # Convertiamo i set in liste ordinate per la stampa
                    people = sorted(list(result['entities']['PER']))
                    orgs = sorted(list(result['entities']['ORG']))
                    locs = sorted(list(result['entities']['LOC']))

                    if people: print(f"   👤 People: {', '.join(people)}")
                    if orgs:   print(f"   🏢 Organisations: {', '.join(orgs)}")
                    if locs:   print(f"   🌍 Locations: {', '.join(locs)}")
                    
                    if not (people or orgs or locs):
                        print("   (No relevant entity found with high confidence)")
                    # -------------------

                    print("\n🔑 KEYWORDS:")
                    top_words = [f"{word} ({freq})" for word, freq in result['keywords']]
                    print(", ".join(top_words))
                    
                    print("\n" + "█"*48)

                    full_text = result['full_text']
                    if full_text and len(full_text) > 0:
                        ask = input("\nDo you want to read the full article? (y/n): ")
                        if ask.lower() == 'y':
                            print("\n--- FULL ARTICLE ---")
                            print(full_text)
                    else:
                        print("\n(Full article not available)")
                else:
                    print("\n❌ Impossible to analyse the article.")
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input.")

if __name__ == "__main__":
    main()
