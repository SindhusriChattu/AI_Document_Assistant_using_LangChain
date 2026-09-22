import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
import re
import tempfile
import os


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.write(
    "Upload a PDF to extract, summarize, and explore "
    "important information from the document."
)


# -----------------------------
# PDF Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name


    try:

        # -----------------------------
        # Load PDF using LangChain
        # -----------------------------

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Combine pages
        full_text = "\n\n".join(
            document.page_content
            for document in documents
        )

        st.success(
            f"Document loaded successfully! "
            f"Pages: {len(documents)}"
        )


        # -----------------------------
        # LangChain Text Splitting
        # -----------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = text_splitter.split_text(full_text)


        st.info(
            f"Document contains approximately "
            f"{len(chunks)} text chunks."
        )


        # -----------------------------
        # Sidebar
        # -----------------------------

        st.sidebar.header("Document Assistant")

        option = st.sidebar.radio(
            "Choose an option",
            [
                "Document Summary",
                "Ask a Question",
                "Key Information",
                "View Extracted Text"
            ]
        )


        # ==================================================
        # DOCUMENT SUMMARY
        # ==================================================

        if option == "Document Summary":

            st.subheader("📋 Document Summary")

            # Split into paragraphs
            paragraphs = [
                p.strip()
                for p in re.split(r"\n\s*\n", full_text)
                if len(p.strip()) > 80
            ]

            # Select important paragraphs
            summary_paragraphs = paragraphs[:8]

            if summary_paragraphs:

                for paragraph in summary_paragraphs:
                    st.write("•", paragraph)

            else:

                st.warning(
                    "Not enough text was found to create a summary."
                )


        # ==================================================
        # QUESTION ANSWERING
        # ==================================================

        elif option == "Ask a Question":

            st.subheader("❓ Ask About the Document")

            question = st.text_input(
                "Enter your question:"
            )

            if st.button("Search Document") and question:

                # Convert question into keywords
                question_words = set(
                    re.findall(
                        r"\b[a-zA-Z]{3,}\b",
                        question.lower()
                    )
                )

                # Score chunks according to keyword matches
                scored_chunks = []

                for chunk in chunks:

                    chunk_words = set(
                        re.findall(
                            r"\b[a-zA-Z]{3,}\b",
                            chunk.lower()
                        )
                    )

                    score = len(
                        question_words.intersection(
                            chunk_words
                        )
                    )

                    if score > 0:
                        scored_chunks.append(
                            (score, chunk)
                        )

                # Sort by relevance
                scored_chunks.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                if scored_chunks:

                    st.success("Relevant information found:")

                    for score, chunk in scored_chunks[:3]:

                        st.write(chunk)
                        st.divider()

                else:

                    st.warning(
                        "No relevant information was found "
                        "in the document."
                    )


        # ==================================================
        # KEY INFORMATION
        # ==================================================

        elif option == "Key Information":

            st.subheader("🔍 Key Information")

            # Extract email
            emails = re.findall(
                r'[\w\.-]+@[\w\.-]+\.\w+',
                full_text
            )

            # Extract phone numbers
            phone_numbers = re.findall(
                r'\b\d{10}\b',
                full_text
            )

            # Extract years
            years = re.findall(
                r'\b(?:19|20)\d{2}\b',
                full_text
            )

            if emails:

                st.write("### 📧 Email")
                for email in set(emails):
                    st.write(email)

            if phone_numbers:

                st.write("### 📱 Phone")
                for phone in set(phone_numbers):
                    st.write(phone)

            if years:

                st.write("### 📅 Years")
                st.write(
                    ", ".join(sorted(set(years)))
                )

            # Word frequency
            words = re.findall(
                r'\b[a-zA-Z]{4,}\b',
                full_text.lower()
            )

            common_words = Counter(words).most_common(15)

            st.write("### 🔑 Frequently Used Words")

            for word, count in common_words:

                st.write(
                    f"**{word}** — {count} occurrences"
                )


        # ==================================================
        # VIEW TEXT
        # ==================================================

        elif option == "View Extracted Text":

            st.subheader("📄 Extracted Document Text")

            st.text_area(
                "Document Content",
                full_text,
                height=500
            )


    except Exception as e:

        st.error(
            f"Unable to process the document: {e}"
        )


    finally:

        # Remove temporary PDF
        if os.path.exists(temp_path):
            os.remove(temp_path)
