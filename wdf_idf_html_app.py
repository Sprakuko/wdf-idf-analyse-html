import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

st.set_page_config(page_title="HTML & WDF*IDF Analyse", layout="wide")
st.title("🔍 HTML- und WDF*IDF-Analyse im Vergleich")

# ==== HTML HEADING CHECKER ====
st.header("1️⃣ HTML Heading Checker")
st.markdown("Bitte gib für jeden Quelltext eine URL und den vollständigen HTML-Code ein.")

urls = []
htmls = []

col1, col2, col3, col4 = st.columns(4)

with col1:
    url1 = st.text_input("🌐 URL zu deinem HTML")
    html1 = st.text_area("Quelltext 1", height=300)
with col2:
    url2 = st.text_input("URL Vergleich 1")
    html2 = st.text_area("Vergleich 1", height=300)
with col3:
    url3 = st.text_input("URL Vergleich 2")
    html3 = st.text_area("Vergleich 2", height=300)
with col4:
    url4 = st.text_input("URL Vergleich 3")
    html4 = st.text_area("Vergleich 3", height=300)

if st.button("🔍 Analysieren"):
    inputs = [(url1, html1), (url2, html2), (url3, html3), (url4, html4)]
    valid_inputs = [(url, html) for url, html in inputs if url.strip() and html.strip()]

    def parse_html_structure(html):
        soup = BeautifulSoup(html, "html.parser")
        meta_title = soup.title.string if soup.title else ""
        meta_description = soup.find("meta", attrs={"name": "description"})
        meta_description = meta_description["content"] if meta_description else ""

        headings = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            headings.append((int(tag.name[1]), f"{tag.name.upper()}: {tag.get_text(strip=True)}", tag.name.lower()))

        styles = []
        h1_count = 0
        for i in range(len(headings)):
            current_level = headings[i][0]
            tag_name = headings[i][2]
            prev_level = headings[i - 1][0] if i > 0 else current_level

            style = ""
            if current_level > prev_level + 1:
                style = "background-color: #ffcdd2"

            if tag_name == "h1":
                h1_count += 1
                if i != 0 or h1_count > 1:
                    style = "background-color: #ffcdd2"

            styles.append(style)

        headings_text = ["→" * (h[0] - 1) + " " + h[1] for h in headings]

        # Nur Text zwischen </header> und </main>
        raw = str(soup)
        try:
            content = raw.split("</header>", 1)[1].split("</main>", 1)[0]
            body_soup = BeautifulSoup(content, "html.parser")
            texts = []
            for tag in body_soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                if tag.name.startswith("h") or tag.name == "p":
                    texts.append(tag.get_text(" ", strip=True))
            body_text = " ".join(texts)
        except:
            body_text = ""

        return headings_text, styles, meta_title, meta_description, body_text

    if len(valid_inputs) < 2:
        st.warning("Bitte gib mindestens zwei vollständige HTML-Quelltexte und ihre zugehörigen URLs ein.")
    else:
        heading_data = {}
        heading_styles = {}
        meta_infos = []
        text_bodies = []
        show_heading_warning = False

        for url, html in valid_inputs:
            headings, styles, meta_title, meta_desc, body_text = parse_html_structure(html)
            heading_data[url] = headings
            heading_styles[url] = styles
            text_bodies.append((url, body_text))
            meta_infos.append({"URL": url, "Meta-Title": meta_title, "Meta-Description": meta_desc})
            if any(s != "" for s in styles):
                show_heading_warning = True

        st.subheader("🔎 Meta-Informationen")
        st.dataframe(pd.DataFrame(meta_infos))

        st.subheader("📑 Überschriftenstruktur im Vergleich")
        if show_heading_warning:
            st.info("Rot markierte Überschriften deuten auf mögliche Fehler hin: Mehr als eine H1-Überschrift, H1 nicht am Anfang oder fehlerhafte Überschriftenhierarchie.")

        max_len = max(len(h) for h in heading_data.values())
        rows = []
        for i in range(max_len):
            row = []
            for url in heading_data:
                text = heading_data[url][i] if i < len(heading_data[url]) else ""
                style = heading_styles[url][i] if i < len(heading_styles[url]) else ""
                row.append(f"<div style='{style}; padding:4px'>{text}</div>" if text else "")
            rows.append(row)
        styled_df = pd.DataFrame(rows, columns=heading_data.keys())
        st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)

        # === WDF*IDF Analysis ===
        st.header("2️⃣ WDF*IDF-Termanalyse")

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

        def clean(text):
            return " ".join([w for w in text.lower().split() if w.isalpha() and w not in stopwords])

        raw_texts = [(url, body) for url, body in text_bodies if body.strip()]
        cleaned_texts = [(url, clean(body)) for url, body in text_bodies if body.strip()]

        if len(raw_texts) < 2:
            st.warning("Mindestens zwei gültige HTML-Quelltexte mit sichtbarem Inhalt erforderlich für die Termanalyse.")
        else:
            vectorizer = CountVectorizer()
            corpus = [text for _, text in raw_texts]
            X = vectorizer.fit_transform(corpus)
            terms = vectorizer.get_feature_names_out()
            freqs = X.toarray()

            word_counts = [sum(f) for f in freqs]
            term_data = []
            for i, (url, raw) in enumerate(raw_texts):
                total_words = word_counts[i]
                term_freq = freqs[i]
                keyword_density = [round((count / total_words) * 100, 3) if total_words > 0 else 0 for count in term_freq]
                df = pd.DataFrame({"Term": terms, "TF": term_freq, "KD (%)": keyword_density})
                df = df[df["Term"].str.lower().apply(lambda x: x.isalpha() and x not in stopwords)]
                df_sorted = df.sort_values("TF", ascending=False).head(20)
                st.subheader(f"🔢 Top-20 Begriffe für {url}")
                st.metric(label="Länge", value=f"{total_words} Wörter")
                st.dataframe(df_sorted.reset_index(drop=True))
