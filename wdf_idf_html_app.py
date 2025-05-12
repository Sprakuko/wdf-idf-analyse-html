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

inputs = [(url1, html1), (url2, html2), (url3, html3), (url4, html4)]

# Funktion zum Parsen von HTML und Ausgabe der H-Tags inkl. Hierarchiekontrolle
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

    headings_text = [h[1] for h in headings]
    return headings_text, styles, soup.title.string if soup.title else "", meta_description

valid_inputs = [(url, html) for url, html in inputs if url.strip() and html.strip()]

if len(valid_inputs) < 2:
    st.warning("Bitte gib mindestens zwei vollständige HTML-Quelltexte und ihre zugehörigen URLs ein.")
else:
    heading_data = {}
    heading_styles = {}
    meta_infos = []

    for url, html in valid_inputs:
        headings, styles, meta_title, meta_desc = parse_html_structure(html)
        heading_data[url] = headings
        heading_styles[url] = styles
        meta_infos.append({"URL": url, "Meta-Title": meta_title, "Meta-Description": meta_desc})

    st.subheader("🔎 Meta-Informationen")
    st.dataframe(pd.DataFrame(meta_infos))

    st.subheader("📑 Überschriftenstruktur im Vergleich")
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

# ==== TERMANALYSE ====
st.header("2️⃣ WDF*IDF Termanalyse")

main_text = html1
comp_texts = [html2, html3, html4]
urls = [url1, url2, url3, url4]
texts = [main_text] + comp_texts
urls = [u for u, t in zip(urls, texts) if u.strip() and t.strip()]
texts = [t for t in texts if t.strip()]

stopwords = set([...])  # gleiche Liste wie oben

def clean(text):
    return " ".join([w for w in text.lower().split() if w.isalpha() and w not in stopwords])

if len(texts) >= 2:
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

    st.subheader("📊 Interaktives Diagramm: Balken = Durchschnitt, Linien = Keyworddichte, Hover = Termfrequenz")
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
        formatted = [f"{term} (KD: {round(df_density[url][term], 2)}%, TF: {df_counts[url][term]})" for term in top_words.index]
        st.markdown(f"**{url}** – Länge: {word_counts[i]} Wörter")
        top_table[url] = formatted
    st.dataframe(top_table)

    st.subheader("📍 Drittelverteilung der Begriffe")
    def split_counts(text, terms):
        words = [w for w in text.lower().split() if w.isalpha() and w not in stopwords]
        thirds = np.array_split(words, 3)
        result = []
        for part in thirds:
            count = pd.Series(part).value_counts()
            result.append([count.get(term, 0) for term in terms])
        return pd.DataFrame(result, index=["Anfang", "Mitte", "Ende"], columns=terms)

    def highlight_max_nonzero(col):
        max_val = col[col != 0].max()
        return ['background-color: #a7ecff' if val == max_val and val != 0 else '' for val in col]

    for i, raw in enumerate(texts[:len(urls)]):
        df_split = split_counts(raw, top_terms)
        st.markdown(f"**{urls[i]}**")
        styled = df_split.style.apply(highlight_max_nonzero, axis=0)
        st.dataframe(styled)
else:
    st.warning("Bitte mindestens zwei HTML-Texte inkl. URL eingeben.")
