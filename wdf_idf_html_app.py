import streamlit as st
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

st.set_page_config(page_title="HTML & WDF*IDF Analyse", layout="wide")
st.title("🔍 HTML- und WDF*IDF-Analyse im Vergleich")

# Abschnitt 1: HTML Input
st.header("1️⃣ HTML Heading Checker")
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

custom_stops = st.text_input("➕ Optional: Eigene Stoppwörter (kommagetrennt)")

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

        body = soup.body
        body_soup = body if body else soup
        texts = [tag.get_text(" ", strip=True) for tag in body_soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])]
        body_text = " ".join(texts)
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

        def format_with_length(text, limit):
            if not text:
                return "-"
            length = len(text)
            if length > limit:
                return f"❗️{text} ({length} Zeichen)"
            elif length > limit - 20:
                return f"🟡 {text} ({length} Zeichen)"
            else:
                return f"🟢 {text} ({length} Zeichen)"

        for info in meta_infos:
            info["Meta-Title"] = format_with_length(info["Meta-Title"], 60)
            info["Meta-Description"] = format_with_length(info["Meta-Description"], 160)

        df_meta = pd.DataFrame(meta_infos)
        st.markdown(f"""
        <div style='overflow-x: auto; width: 100%;'>
            <style>
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px 12px; text-align: left; white-space: nowrap; }}
            </style>
            {df_meta.to_html(escape=False, index=False)}
        </div>
        """, unsafe_allow_html=True)

        # Bereinigung, TF/KD-Berechnung etc.
        stopwords_raw = """aber, alle, als, am, an, auch, auf, aus, bei, bin, bis, bist, da, damit, dann,
        der, die, das, dass, deren, dessen, dem, den, denn, dich, dir, du, ein, eine,
        einem, einen, einer, eines, er, es, etwas, euer, eure, für, gegen, gehabt, hab,
        habe, haben, hat, hier, hin, hinter, ich, ihm, ihn, ihnen, ihr, ihre, im, in,
        ist, jede, jedem, jeden, jeder, jedes, jener, jenes, jetzt, kann, kein, keine,
        keinem, keinen, keiner, keines, mich, mir, mit, muss, müssen, nach, nein, nicht,
        nichts, noch, nun, nur, ob, oder, ohne, sehr, sein, seine, seinem, seinen, seiner,
        seines, sie, sind, so, soll, sollen, sollte, sonst, um, und, uns, unser, unter,
        viel, vom, von, vor, war, waren, warst, was, weiter, welche, welchem, welchen,
        welcher, welches, wenn, wer, werde, werden, werdet, weshalb, wie, wieder, will, wir,
        wird, wirst, wo, wollen, wollte, würde, würden, zu, zum, zur, über"""
        stopwords = set(w.strip().lower() for w in re.split(r",\s*", stopwords_raw))
        if custom_stops:
            user_stops = set(w.strip().lower() for w in custom_stops.split(",") if w.strip())
            stopwords.update(user_stops)

        urls = [u for u, t in text_bodies if t.strip()]
        raw_texts = [t for _, t in text_bodies if t.strip()]
        raw_word_counts = [len(re.findall(r"\\b\\w+\\b", t)) for t in raw_texts]

        def clean_text(text, stopwords):
            tokens = re.findall(r'\\b[\\wäöüß]+\\b', text.lower(), flags=re.UNICODE)
            return " ".join([w for w in tokens if w not in stopwords])

        cleaned_texts = [clean_text(t, stopwords) for t in raw_texts]
        clean_word_counts = [len(t.split()) for t in cleaned_texts]

        # Prüfen, ob alle bereinigten Texte leer sind
        if all(text.strip() == "" for text in cleaned_texts):
            st.error("❌ Nach dem Entfernen der Stoppwörter enthalten alle Texte keine auswertbaren Begriffe mehr.")
            st.stop()
        
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(cleaned_texts)
        terms = vectorizer.get_feature_names_out()
        df_counts = pd.DataFrame(matrix.toarray(), columns=terms, index=urls).T
        df_density = df_counts.copy()
        for i, label in enumerate(urls):
            df_density[label] = (df_counts[label] / clean_word_counts[i] * 100).round(2)

        # === Benutzerdefinierter Begriff ===
        st.subheader("🔎 Manuelle Analyse eines Begriffs")
        keyword_input = st.text_input("Gib einen Begriff ein (z. B. 'surfcamps'):")
        if st.button("Neu berechnen"):
            if keyword_input:
                term = keyword_input.strip().lower()
                data = []
                for url in urls:
                    tf = df_counts.at[term, url] if term in df_counts.index else 0
                    kd = df_density.at[term, url] if term in df_density.index else 0.0
                    data.append({"URL": url, "TF": tf, "Keyworddichte (%)": round(kd, 2)})

                df_keyword = pd.DataFrame(data)
                st.markdown(f"**Auswertung für:** `{term}`")
                st.table(df_keyword)
            else:
                st.warning("Bitte gib einen Begriff ein.")

        st.subheader("📊 Interaktives Diagramm")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_terms, y=avg_top, name="Durchschnitt", marker_color="lightgray"))

        for label in urls:
            if label in df_top_density.columns:
                tf_values = [f"TF: {df_top_counts.at[term, label]}" if term in df_top_counts.index else "TF: 0" for term in top_terms]
                y_values = [df_top_density.at[term, label] if term in df_top_density.index else 0 for term in top_terms]
                fig.add_trace(go.Scatter(x=top_terms, y=y_values, mode='lines+markers', name=label, text=tf_values, hoverinfo='text+y'))

        fig.update_layout(
            height=500,
            width=1600,
            xaxis=dict(title="Top 50 Begriffe (nach durchschnittlicher Keyworddichte)", tickangle=45),
            yaxis=dict(title="Keyworddichte (%)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏅 Top-20 Begriffe je Text (mit KD + TF)")
        top_table = pd.DataFrame(index=range(1, 21))
        for i, url in enumerate(urls):
            top_words = df_density[url].sort_values(ascending=False).head(20)
            formatted = [f"{term} (KD: {round(df_density.at[term, url], 2)}%, TF: {df_counts.at[term, url]})" for term in top_words.index]
            st.markdown(f"**{url}** – Länge: {raw_word_counts[i]} Wörter (bereinigt: {clean_word_counts[i]})")
            top_table[url] = formatted
        st.dataframe(top_table)

        st.subheader("📍 Drittelverteilung der Begriffe")

        def split_counts(cleaned_text, terms):
            words = cleaned_text.split()
            thirds = np.array_split(words, 3)
            result = []
            for part in thirds:
                count = pd.Series(part).value_counts()
                result.append([count.get(term, 0) for term in terms])
            return pd.DataFrame(result, index=["Anfang", "Mitte", "Ende"], columns=terms)

        def highlight_max_nonzero(col):
            max_val = col[col != 0].max()
            return ['background-color: #a7ecff' if val == max_val and val != 0 else '' for val in col]

        for i, url in enumerate(urls):
            df_split = split_counts(cleaned_texts[i], top_terms)
            st.markdown(f"**{url}**")
            styled = df_split.style.apply(highlight_max_nonzero, axis=0)
            st.dataframe(styled)
