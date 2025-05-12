import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

st.set_page_config(page_title="HTML Heading Check", layout="wide")
st.title("📋 H-Tags & Meta-Daten Analyse")

col1, col2, col3, col4 = st.columns(4)
html1 = col1.text_area("🔍 Quelltext – Dein HTML", height=300)
html2 = col2.text_area("Vergleichstext 1", height=300)
html3 = col3.text_area("Vergleichstext 2", height=300)
html4 = col4.text_area("Vergleichstext 3", height=300)

html_list = [html1, html2, html3, html4]
labels = ["Dein HTML", "Vergleich 1", "Vergleich 2", "Vergleich 3"]

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

# Auswertung
for i, html in enumerate(html_list):
    if html.strip():
        h1_text, meta_title, meta_desc, headings, styles = parse_html_structure(html)
        st.markdown(f"### 🧾 {labels[i]}")
        st.markdown(f"**H1-Titel:** {h1_text}")
        st.markdown(f"**Meta-Title:** {meta_title}")
        st.markdown(f"**Meta-Description:** {meta_desc}")

        df = pd.DataFrame({"Überschriften": headings, "Fehler": ["🔴" if s else "" for s in styles]})

        def to_colored_html(df):
            html = "<table style='border-collapse: collapse; width: 100%;'>"
            html += "<tr><th style='border: 1px solid #ddd; padding: 8px;'>Fehler</th><th style='border: 1px solid #ddd; padding: 8px;'>Überschrift</th></tr>"
            for _, row in df.iterrows():
                bg = " style='background-color: #ffcdd2;'" if row["Fehler"] else ""
                html += f"<tr{bg}><td style='border: 1px solid #ddd; padding: 8px;'>{row['Fehler']}</td><td style='border: 1px solid #ddd; padding: 8px;'>{row['Überschriften']}</td></tr>"
            html += "</table>"
            return html

        st.markdown(to_colored_html(df), unsafe_allow_html=True)
        st.markdown("---")
