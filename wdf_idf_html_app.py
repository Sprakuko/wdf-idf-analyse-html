import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import difflib

st.set_page_config(page_title="WDF*IDF HTML Analyse", layout="wide")
st.title("🔍 WDF*IDF + H-Tag Vergleich aus HTML-Quelltexten")

main_html = st.text_area("📝 HTML-Quelltext deines Textes", height=200)
main_url = st.text_input("🌐 URL zu deinem Text")

col1, col2, col3 = st.columns(3)
comp1 = col1.text_area("HTML-Vergleichstext 1", height=150)
url1 = col1.text_input("URL Vergleich 1")
comp2 = col2.text_area("HTML-Vergleichstext 2", height=150)
url2 = col2.text_input("URL Vergleich 2")
comp3 = col3.text_area("HTML-Vergleichstext 3", height=150)
url3 = col3.text_input("URL Vergleich 3")

texts_raw = [main_html, comp1, comp2, comp3]
urls = [main_url, url1, url2, url3]
texts_raw = [t for t in texts_raw if t.strip()]
urls = [u for u in urls if u.strip()]

stopwords = set([
    "aber", "alle", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann",
    "der", "die", "das", "dass", "deren", "dessen", "dem", "den", "denn", "dich", "dir", "du", "ein", "eine",
    "einem", "einen", "einer", "eines", "er", "es", "etwas", "euer", "eure", "für", "gegen", "gehabt", "hab",
    "habe", "haben", "hat", "hier", "hin", "hinter", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "im", "in",
    "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jener", "jenes", "jetzt", "kann", "kein", "keine",
    "keinem", "keinen", "keiner", "keines", "mich", "mir", "mit", "muss", "müssen", "nach", "nein", "nicht",
    "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner",
    "seines", "sie", "sind", "so", "soll", "sollen", "sollte", "sonst", "um", "und", "uns", "unser", "unter",
    "viel", "vom", "von", "vor", "war", "waren", "warst", "was", "weiter", "welche", "welchem", "welchen",
    "welcher", "welches", "wenn", "wer", "werde", "werden", "werdet", "weshalb", "wie", "wieder", "will", "wir",
    "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu", "zum", "zur", "über"
])

def extract_text_and_headings(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    headings = {}
    for i in range(1, 7):
        tag = f"h{i}"
        headings[tag] = [h.get_text().strip() for h in soup.find_all(tag)]
    return text, headings

def clean(text):
    return " ".join([w for w in text.lower().split() if w.isalpha() and w not in stopwords])

if st.button("🔍 Analyse starten"):
    if len(texts_raw) < 2 or len(urls) < 2:
        st.warning("Bitte gib mindestens zwei HTML-Quelltexte mit zugehörigen URLs ein.")
    else:
        texts = []
        headings_list = []
        for html in texts_raw:
            plain_text, headings = extract_text_and_headings(html)
            texts.append(plain_text)
            headings_list.append(headings)

        # Headline-Vergleich anzeigen
        st.subheader("🧠 Übersicht: H1-H6 Überschriftenstruktur")
        all_rows = []
        for i, url in enumerate(urls):
            row = {}
            for h in range(1, 7):
                tag = f"h{h}"
                row[tag] = " | ".join(headings_list[i][tag]) if tag in headings_list[i] else ""
            row["URL"] = url
            all_rows.append(row)
        df_headlines = pd.DataFrame(all_rows).set_index("URL")
        st.dataframe(df_headlines)

        # Ähnlichkeit farblich markieren
        st.subheader("🎯 Semantisch ähnliche Begriffe (aus H-Tags)")
        all_terms = []
        for h_dict in headings_list:
            for tags in h_dict.values():
                all_terms.extend(tags)
        matches = set()
        for term in all_terms:
            for other in all_terms:
                if term != other and difflib.SequenceMatcher(None, term, other).ratio() > 0.75:
                    matches.add(term)
                    matches.add(other)
        if matches:
            st.markdown("**Ähnliche Begriffe (Überschriften):**")
            st.markdown(", ".join(sorted(matches)))
        else:
            st.info("Keine stark ähnlichen Begriffe in den H-Tags gefunden.")

        # WDF*IDF Analyse
        st.subheader("📊 WDF*IDF Analyse auf sichtbarem HTML-Text")
        cleaned = [clean(t) for t in texts]
        word_counts = [len(t.split()) for t in cleaned]
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(cleaned)
        terms = vectorizer.get_feature_names_out()
        df_counts = pd.DataFrame(matrix.toarray(), columns=terms, index=urls).T
        df_density = df_counts.copy()

        for i, label in enumerate(urls):
            df_density[label] = (df_counts[label] / word_counts[i] * 100).round(2)

        df_avg = df_density.mean(axis=1)
        top_terms = df_avg.sort_values(ascending=False).head(50).index
        df_top_density = df_density.loc[top_terms]
        df_top_counts = df_counts.loc[top_terms]
        avg_top = df_top_density.mean(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_terms, y=avg_top, name="Durchschnitt", marker_color="lightgray"))
        for label in urls:
            fig.add_trace(go.Scatter(
                x=top_terms,
                y=df_top_density[label],
                mode='lines+markers',
                name=label,
                text=[f"TF: {df_top_counts[label][term]}" for term in top_terms],
                hoverinfo='text+y'
            ))
        fig.update_layout(
            height=500,
            width=1600,
            xaxis=dict(title="Top 50 Begriffe", tickangle=45),
            yaxis=dict(title="Keyworddichte (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)