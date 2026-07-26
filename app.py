import os
import streamlit as st
from ingest import build_vectorstore
from graph import ask

st.set_page_config(page_title="CorrectiveRAG", page_icon="🔍")
st.title("🔍CorrectiveRAG")
st.caption("Upload a PDF. If your document doesn't have answer. I will check the live web instead.")

#store the chat history so it doesn't reset every time
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# upload the pdf
uploaded_file= st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    save_path= os.path.join("uploads", uploaded_file.name)
    with open (save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Index this document"):
        with st.spinner("Reading and indexing document......"):
            build_vectorstore(save_path)
        st.success("Document indexed! You can ask question now.")

st.divider()

# show old question and answer
for entry in st.session_state.chat_history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Assistant:** {entry['answer']}")
    st.divider()

#ask a new question
question= st.text_input("Ask your question")
if st.button("Ask")and question:
    with st.spinner("Thinking..(retrieving, grading, verifying)......"):
        result = ask(question, chat_history=st.session_state.chat_history)

    st.subheader("Answer")
    st.write(result["answer"])
    # show the extra detailed about how the answer was made
    with st.expander("How I get this answer"):
        st.write(f"**Rewritten question:** {result['question']}")
        st.write(f"**Source used:** {result['source']}")
        st.write(f"**Retries needed:** {result['retries']}")
        st.write(f"**Sub-questions:** {result['sub_questions']}")

    # save the Q&A so the next questions remembers it
    st.session_state.chat_history.append({
        "question": question,
        "answer": result["answer"]
    })