import os
import sys
import logging
import warnings
import requests
import time
from collections import Counter
from newsapi import NewsApiClient

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class NewsAssistant:
    def __init__(self, api_key):
        t_start = time.time()
        print("--- Initialising NewsAssistant ---")
        self.newsapi = NewsApiClient(api_key=api_key)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        self._device = None
        self._sentiment_analyzer = None
        self._summarizer = None
        self._ner_pipeline = None
        self._nltk_ready = False
        print(f"   ⏱️  Init finished in {time.time() - t_start:.2f}s")

    @property
    def device(self):
        if self._device is None:
            self._device = -1
            print("💻 Using CPU")
        return self._device

    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            t_load = time.time()
            print("⏳ Loading Sentiment Model... (this happens only once)")
            from transformers import pipeline, logging as hf_logging
            hf_logging.set_verbosity_error()
            self._sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=self.device)
            print(f"   ⏱️  Sentiment Model loaded in {time.time() - t_load:.2f}s")
        return self._sentiment_analyzer

    @property
    def summarizer(self):
        if self._summarizer is None:
            t_load = time.time()
            print("⏳ Loading Summarization Model... (this happens only once)")
            from transformers import pipeline, logging as hf_logging
            hf_logging.set_verbosity_error()
            self._summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=self.device)
            print(f"   ⏱️  Summarization Model loaded in {time.time() - t_load:.2f}s")
        return self._summarizer

    @property
    def ner_pipeline(self):
        if self._ner_pipeline is None:
            t_load = time.time()
            print("⏳ Loading NER Model... (this happens only once)")
            from transformers import pipeline, logging as hf_logging
            hf_logging.set_verbosity_error()
            self._ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=self.device)
            print(f"   ⏱️  NER Model loaded in {time.time() - t_load:.2f}s")
        return self._ner_pipeline

    def _ensure_nltk(self):
        if not self._nltk_ready:
            t_nltk = time.time()
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Downloading NLTK resources...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            self._nltk_ready = True
            print(f"   ⏱️  NLTK check/download finished in {time.time() - t_nltk:.2f}s")

    def get_top_headlines(self, category='general', page_size=20):
        t_api = time.time()
        try:
            top_headlines = self.newsapi.get_top_headlines(
                category=category, language='en', page_size=page_size
            )
            print(f"   ⏱️  NewsAPI call finished in {time.time() - t_api:.2f}s")
            return top_headlines.get('articles', [])
        except Exception as e:
            print(f"API error: {e}")
            return []

    def preprocess_text(self, text):
        self._ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        if not text: return []
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        custom_stops = {'news', 'report', 'says', 'new', 'update', 'live', 'said', 'us', 'could', 'would'}
        stop_words.update(custom_stops)
        filtered = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
        return filtered

    def generate_daily_briefing(self, category='general'):
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud

        print(f"\n--- 📊 GENERATING DAILY PRESS REVIEW ({category.upper()}) ---")
        print("Downloading and analysing news...")
        
        articles = self.get_top_headlines(category=category, page_size=30)
        
        if not articles:
            print("News not found.")
            return []

        all_text = ""
        sentiments = []
        
        t_sent_loop = time.time()
        analyzer = self.sentiment_analyzer 

        for art in articles:
            content = f"{art['title']} {art['description'] or ''}" 
            all_text += content + " "
            try:
                res = analyzer(art['title'][:512])[0]
                sentiments.append(res['label'])
            except:
                pass
        
        print(f"   ⏱️  Batch Sentiment Analysis finished in {time.time() - t_sent_loop:.2f}s")

        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        if total > 0:
            pos_pct = (sentiment_counts['POSITIVE'] / total) * 100
            neg_pct = (sentiment_counts['NEGATIVE'] / total) * 100
            sentiment_summary = f"Global Sentiment: {pos_pct:.1f}% POSITIVE | {neg_pct:.1f}% NEGATIVE"
        else:
            sentiment_summary = "Impossible to calculate sentiment "

        print("Generating Word Cloud...")
        t_wc = time.time()
        clean_tokens = self.preprocess_text(all_text)
        clean_text_string = " ".join(clean_tokens)
        
        if not clean_text_string:
            print("Not enough text to generate a Word Cloud.")
            return articles

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_text_string)
        print(f"   ⏱️  WordCloud generated in {time.time() - t_wc:.2f}s")

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"TOPICS OF THE DAY - {category.upper()}\n{sentiment_summary}", fontsize=14, color='darkblue')
        
        print(f"\n>>> {sentiment_summary}")
        print(">>> A Word Cloud window is open. Close to continue.")
        plt.show()
        
        return articles
    
    def _run_ner_on_full_text(self, text, chunk_size=400):
        entities = {"PER": set(), "ORG": set(), "LOC": set()}
        if not text: return entities

        ner_model = self.ner_pipeline
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        for chunk in chunks:
            try:
                ner_results = ner_model(chunk)
                for entity in ner_results:
                    if entity['score'] < 0.85: continue
                    word = entity['word'].strip()
                    if word.startswith("##") or len(word) < 4: continue
                    group = entity['entity_group']
                    if group in entities:
                        entities[group].add(word)
            except Exception:
                continue
        return entities

    def analyze_single_article(self, url):
        import trafilatura
        
        blocked_domains = ['bloomberg.com', 'wsj.com', 'ft.com', 'nytimes.com', 'washingtonpost.com', 'medium.com']
        if any(domain in url for domain in blocked_domains):
            print("   🚫 Skipped (Protected/Paywall domain)")
            return None

        t_total = time.time()
        try:
            print(f"   Trying smart download from: {url[:40]}...")
            
            t_down = time.time()
            full_text = None
            
            try:
                response = requests.get(url, headers=self.headers, timeout=(6, 10))
                response.raise_for_status() 
                full_text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
            except Exception as e:
                print(f"   ⚠️ Download failed or timed out: {e}")
                return None
            
            print(f"   ⏱️  Download & Extraction finished in {time.time() - t_down:.2f}s")
                    
            if not full_text or len(full_text) < 100: 
                print("   ⚠️ Text too short or empty.")
                return None

            input_len = len(full_text.split())
            max_len = min(130, input_len // 2)
            
            t_sum = time.time()
            try:
                summary = self.summarizer(full_text[:3000], max_length=max_len, min_length=30, truncation=True)[0]['summary_text']
            except:
                summary = "Summary generation failed."
            print(f"   ⏱️  Summarization finished in {time.time() - t_sum:.2f}s")
            
            t_sent = time.time()
            sentiment = self.sentiment_analyzer(full_text[:512])[0]
            print(f"   ⏱️  Sentiment Analysis finished in {time.time() - t_sent:.2f}s")
            
            clean_tokens = self.preprocess_text(full_text)
            keywords = Counter(clean_tokens).most_common(5)

            print("   Processing NER on full text...")
            t_ner = time.time()
            entities = self._run_ner_on_full_text(full_text)
            print(f"   ⏱️  NER finished in {time.time() - t_ner:.2f}s")
            
            print(f"   ⏱️  TOTAL ANALYSIS TIME: {time.time() - t_total:.2f}s")

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

                    print("\n🔍 NER ANALYSIS:")
                    people = sorted(list(result['entities']['PER']))
                    orgs = sorted(list(result['entities']['ORG']))
                    locs = sorted(list(result['entities']['LOC']))

                    if people: print(f"   👤 People: {', '.join(people)}")
                    if orgs:   print(f"   🏢 Organisations: {', '.join(orgs)}")
                    if locs:   print(f"   🌍 Locations: {', '.join(locs)}")
                    
                    if not (people or orgs or locs):
                        print("   (No relevant entity found with high confidence)")

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
