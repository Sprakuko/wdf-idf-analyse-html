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
cols = st.columns(4)
urls = []
htmls = []
for i in range(4):
    urls.append(cols[i].text_input(f"🌐 URL {i+1}" if i == 0 else f"URL Vergleich {i}"))
    htmls.append(cols[i].text_area(f"Quelltext {i+1}" if i == 0 else f"Vergleich {i}", height=300))

custom_stops = st.text_input("➕ Optional: Eigene Stoppwörter (kommagetrennt)")

# Zusätzliche Code-spezifische Stopwords
code_stopwords = {
    "var", "return", "if", "else", "function", "new", "go", "static"
}

# Hilfsfunktionen

def parse_html_structure(html):
    soup = BeautifulSoup(html, "html.parser")
    meta_title = soup.title.string.strip() if soup.title and soup.title.string else ""
    meta_description_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_description_tag["content"].strip() if meta_description_tag and "content" in meta_description_tag.attrs else ""

    headings, styles = [], []
    h1_count = 0
    prev_level = 0

    for tag in soup.find_all([f"h{i}" for i in range(1, 7)]):
        level = int(tag.name[1])
        style = ""
        if level > prev_level + 1:
            style = "background-color: #ffcdd2"
        if tag.name == "h1":
            h1_count += 1
            if len(headings) > 0 or h1_count > 1:
                style = "background-color: #ffcdd2"
        headings.append((level, f"{tag.name.upper()}: {tag.get_text(strip=True)}"))
        styles.append(style)
        prev_level = level

    headings_text = ["→" * (level - 1) + " " + text for level, text in headings]

    body_soup = soup.body if soup.body else soup
    text = " ".join(tag.get_text(" ", strip=True) for tag in body_soup.find_all(["p"] + [f"h{i}" for i in range(1, 7)]))

    return headings_text, styles, meta_title, meta_description, text

def format_with_length(text, limit):
    if not text:
        return "-"
    length = len(text)
    if length > limit:
        return f"❗️{text} ({length} Zeichen)"
    elif length > limit - 20:
        return f"🟡 {text} ({length} Zeichen)"
    return f"🟢 {text} ({length} Zeichen)"

if st.button("🔍 Analysieren"):
    valid_inputs = [(url, html) for url, html in zip(urls, htmls) if url.strip() and html.strip()]

    if len(valid_inputs) < 2:
        st.warning("Bitte gib mindestens zwei vollständige HTML-Quelltexte und ihre zugehörigen URLs ein.")
    else:
        heading_data, heading_styles, meta_infos, text_bodies = {}, {}, [], []

        for url, html in valid_inputs:
            headings, styles, meta_title, meta_desc, body_text = parse_html_structure(html)
            heading_data[url] = headings
            heading_styles[url] = styles
            text_bodies.append((url, body_text))
            meta_infos.append({
                "URL": url,
                "Meta-Title": format_with_length(meta_title, 60),
                "Meta-Description": format_with_length(meta_desc, 160)
            })

        st.subheader("🔎 Meta-Informationen")
        st.dataframe(pd.DataFrame(meta_infos))

        st.subheader("📑 Überschriftenstruktur im Vergleich")
        max_len = max(len(hs) for hs in heading_data.values())
        heading_rows = []
        for i in range(max_len):
            row = []
            for url in heading_data:
                row.append(heading_data[url][i] if i < len(heading_data[url]) else "")
            heading_rows.append(row)
        heading_df = pd.DataFrame(heading_rows, columns=heading_data.keys())
        st.markdown(f"""
        <div style='overflow-x:auto;'>
        {heading_df.to_html(escape=False, index=False)}
        </div>
        """, unsafe_allow_html=True)

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
        stopwords.update(code_stopwords)
        if custom_stops:
            user_stops = set(w.strip().lower() for w in custom_stops.split(",") if w.strip())
            stopwords.update(user_stops)

        urls = [u for u, t in text_bodies if t.strip()]
        raw_texts = [t for _, t in text_bodies if t.strip()]
        raw_word_counts = [len(re.findall(r"\b\w+\b", t)) for t in raw_texts]

        def clean_text(text, stopwords):
            tokens = re.findall(r'\b[\wäöüß]+\b', text.lower(), flags=re.UNICODE)
            return " ".join([w for w in tokens if w not in stopwords])

        cleaned_texts = [clean_text(t, stopwords) for t in raw_texts]
        clean_word_counts = [len(t.split()) for t in cleaned_texts]

        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(cleaned_texts)
        terms = vectorizer.get_feature_names_out()
        df_counts = pd.DataFrame(matrix.toarray(), columns=terms, index=urls).T
        df_density = df_counts.copy()
        for i, label in enumerate(urls):
            df_density[label] = (df_counts[label] / clean_word_counts[i] * 100).round(2)

        df_avg = df_density.mean(axis=1)
        top_terms = df_avg.sort_values(ascending=False).head(50).index
        df_top_density = df_density.loc[top_terms]
        df_top_counts = df_counts.loc[top_terms]
        avg_top = df_top_density.mean(axis=1)

        st.subheader("📊 Interaktives Diagramm")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_terms, y=avg_top, name="Durchschnitt", marker_color="lightgray"))

        for label in urls:
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

        st.download_button(
            label="📥 Excel-Download der Keyword-Dichte",
            data=df_top_density.to_excel(index=True),
            file_name="keyword_dichte.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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
