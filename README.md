📰 AI News Assistant & Press Review
AI News Assistant è uno strumento CLI (Command Line Interface) avanzato scritto in Python che aggrega notizie in tempo reale e le analizza utilizzando modelli di Intelligenza Artificiale (NLP).

Non si limita a scaricare le notizie: crea riassunti automatici, analizza il "sentiment" degli articoli, estrae le entità nominate (persone, luoghi, aziende) e genera visualizzazioni grafiche (Word Cloud) dei temi caldi del giorno.

✨ Funzionalità Principali
🌐 Aggregazione Notizie: Scarica le ultime notizie da fonti affidabili in lingua inglese tramite NewsAPI.

📊 Analisi del Sentiment: Determina se il tono di un articolo o di una rassegna stampa è Positivo o Negativo.

🧠 Riassunti Automatici: Utilizza reti neurali per leggere articoli lunghi e generarne riassunti concisi.

🔍 Named Entity Recognition (NER): Identifica ed estrae automaticamente i protagonisti delle notizie (Persone, Organizzazioni, Luoghi).

☁️ Word Cloud: Genera una nuvola di parole visiva basata sulla frequenza dei termini negli articoli del giorno.

📥 Smart Scraping: Estrae il testo completo dagli articoli (bypassando layout complessi) utilizzando trafilatura.

🛡️ Filtro Paywall: Esclude automaticamente domini noti per avere paywall rigidi (es. WSJ, Bloomberg) per evitare errori di analisi.

🛠️ Tecnologie e Modelli
Il progetto sfrutta potenti librerie open-source e modelli pre-addestrati di Hugging Face:

NewsAPI: Per il recupero dei metadati delle notizie.

Transformers (Hugging Face):

Sentiment: distilbert-base-uncased-finetuned-sst-2-english

Summarization: sshleifer/distilbart-cnn-12-6

NER: dslim/bert-base-NER

NLTK: Per la pulizia del testo, tokenizzazione e rimozione delle stop-words.

Trafilatura: Per lo scraping efficiente del contenuto web.

WordCloud & Matplotlib: Per la generazione e visualizzazione dei grafici.

🚀 Installazione
Prerequisiti
Assicurati di avere Python 3.8+ installato.

⚙️ Configurazione
Per far funzionare il programma, hai bisogno di una API Key gratuita di NewsAPI.

Registrati su NewsAPI.org.

Apri il file main.py.

Cerca la variabile API_KEY nella funzione main() e inserisci la tua chiave.

🖥️ Utilizzo
Avvia il programma da terminale

Flusso di lavoro:
All'avvio, il sistema caricherà i modelli AI (la prima volta potrebbe richiedere qualche minuto per il download).

Seleziona una categoria (General, Technology, Business, ecc.).

Verrà mostrata una Word Cloud e un riassunto del Sentiment Globale della rassegna stampa.

Chiudi la finestra del grafico per vedere la lista degli articoli.

Seleziona il numero di un articolo per analizzarlo in profondità (Riassunto, NER, Sentiment dettagliato).

📂 Struttura del Codice
NewsAssistant: La classe principale. Gestisce il caricamento "lazy" (ritardato) dei modelli pesanti per velocizzare l'avvio iniziale.

generate_daily_briefing: Scarica i titoli, calcola il sentiment medio e crea la Word Cloud.

analyze_single_article: Scarica il testo completo di un URL specifico ed esegue la pipeline di analisi profonda.

⚠️ Note
Performance: L'analisi AI è intensiva per la CPU. Se hai una GPU NVIDIA configurata con CUDA, i modelli transformers proveranno ad usarla, altrimenti useranno la CPU (più lento).

Traffico Dati: Il download dei modelli di Hugging Face richiede diverse centinaia di MB al primo avvio.

📄 Licenza
Distribuito sotto licenza MIT.
