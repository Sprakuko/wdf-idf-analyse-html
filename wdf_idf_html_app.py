import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

st.set_page_config(page_title="HTML Heading Check", layout="wide")
st.title("📋 H-Tags & Meta-Daten Analyse im Vergleich")

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
    h1_tag = soup.find("h1")
    h1_text = h1_tag.get_text(strip=True) if h1_tag else ""

    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        headings.append((int(tag.name[1]), f"{tag.name.upper()}: {tag.get_text(strip=True)}"))

    styles = []
    for i in range(len(headings)):
        current_level = headings[i][0]
        prev_level = headings[i - 1][0] if i > 0 else current_level
        if current_level > prev_level + 1:
            styles.append("background-color: #ffcdd2")
        else:
            styles.append("")

    headings_text = [h[1] for h in headings]
    return h1_text, meta_title, meta_description, headings_text, styles

# Ergebnisse sammeln
valid_inputs = [(url, html) for url, html in inputs if url.strip() and html.strip()]

if len(valid_inputs) < 2:
    st.warning("Bitte gib mindestens zwei vollständige HTML-Quelltexte und ihre zugehörigen URLs ein.")
else:
    h1_list = []
    title_list = []
    desc_list = []
    headings_dict = {}
    error_marks = {}

    for url, html in valid_inputs:
        h1, title, desc, headings, styles = parse_html_structure(html)
        urls.append(url)
        h1_list.append(h1)
        title_list.append(title)
        desc_list.append(desc)
        headings_dict[url] = headings
        error_marks[url] = styles

    # Meta-Infos anzeigen
    meta_df = pd.DataFrame({
        "URL": urls,
        "H1-Titel": h1_list,
        "Meta-Title": title_list,
        "Meta-Description": desc_list
    })
    st.subheader("🔎 Meta-Informationen & H1")
    st.dataframe(meta_df)

    # Überschriftenvergleich nebeneinander
    st.subheader("📑 Überschriftenstruktur im Vergleich (Reihenfolge & Hierarchiefehler)")
    max_len = max(len(v) for v in headings_dict.values())
    styled_rows = []
    for i in range(max_len):
        row = []
        for url in urls:
            heading = headings_dict[url][i] if i < len(headings_dict[url]) else ""
            style = error_marks[url][i] if i < len(error_marks[url]) else ""
            if heading:
                cell = f"<div style='{style}; padding:4px'>{heading}</div>"
            else:
                cell = ""
            row.append(cell)
        styled_rows.append(row)

    styled_df = pd.DataFrame(styled_rows, columns=urls)
    st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)
