import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Assistant")
st.write(
    "Upload a PDF and use AI to summarize the document "
    "or ask questions about its content."
)


# API Key
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key is not configured.")
    st.stop()


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.2
)


# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name


    # Load PDF
    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        full_text = "\n\n".join(
            doc.page_content for doc in documents
        )

        st.success(
            f"Document loaded successfully! "
            f"Pages: {len(documents)}"
        )

    except Exception as e:
        st.error(f"Error loading document: {e}")
        st.stop()


    # Sidebar
    st.sidebar.header("Document Actions")

    action = st.sidebar.radio(
        "Choose an action:",
        [
            "Summarize Document",
            "Ask a Question"
        ]
    )


    # -------------------------
    # SUMMARY
    # -------------------------

    if action == "Summarize Document":

        if st.button("Generate Summary"):

            with st.spinner("Generating summary..."):

                prompt = ChatPromptTemplate.from_template(
                    """
                    You are an AI document assistant.

                    Summarize the following document clearly.

                    Provide:
                    1. Short overview
                    2. Important points
                    3. Key findings
                    4. Important dates or information
                    5. Conclusion

                    Document:
                    {document}
                    """
                )

                chain = prompt | llm

                response = chain.invoke({
                    "document": full_text
                })

                st.subheader("📋 Document Summary")
                st.write(response.content)


    # -------------------------
    # QUESTION ANSWERING
    # -------------------------

    else:

        question = st.text_input(
            "Ask a question about the document:"
        )

        if st.button("Get Answer") and question:

            with st.spinner("Finding the answer..."):

                prompt = ChatPromptTemplate.from_template(
                    """
                    You are an AI document assistant.

                    Answer the user's question using only
                    the information available in the document.

                    If the answer is not available in the
                    document, say:
                    "The information is not available
                    in the uploaded document."

                    Document:
                    {document}

                    User Question:
                    {question}
                    """
                )

                chain = prompt | llm

                response = chain.invoke({
                    "document": full_text,
                    "question": question
                })

                st.subheader("💡 Answer")
                st.write(response.content)


    # Remove temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
