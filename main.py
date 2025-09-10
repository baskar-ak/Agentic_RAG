import os
import dspy
import time
import streamlit as st
import tempfile

from file_loader import FileLoader
from chunker import Chunker
from retriever import RetrieverAgent
from utils import Utils

class RAGApp:
	"""
	AI Research Assistant Application.
	"""
	def __init__(self):
		self.loader = FileLoader()
		self.chunker = Chunker()
		self.retriever = RetrieverAgent()
		self.utils = Utils()
		if not dspy.settings.lm:
			dspy.configure(lm=dspy.LM(f"openai/{self.utils.llm_model}", api_key=self.utils.api_key)) # Configure LM for DSPy

	def main(self):
		st.set_page_config(page_title="AI Research Assistant", layout="wide")
		st.title("How can I assist you today? 🤖")
		st.subheader("Simply upload a document and ask anything!")

		# Initialize session state
		if "extracted_text" not in st.session_state:
			st.session_state.extracted_text = None
		if "chunks_stored" not in st.session_state:
			st.session_state.chunks_stored = False
		if "user_query" not in st.session_state:
			st.session_state.user_query = None
		if "reranked_results" not in st.session_state:
			st.session_state.reranked_results = None
		if "answer" not in st.session_state:
			st.session_state.answer = None

		# File / URL upload
		uploaded_file = st.file_uploader("Upload a File:", type=["pdf", "docx", "txt"])
		url_input = st.text_input("Or Enter a web article URL:")

		# Extract content
		if not st.session_state.extracted_text and (uploaded_file or url_input):
			with st.spinner("Processing..."):
				try:
					if uploaded_file:
						with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
							tmp.write(uploaded_file.read())
							tmp_path = tmp.name
						text = self.loader.load(tmp_path)
						os.remove(tmp_path)
					else:
						text = self.loader.load(url_input)

					st.session_state.extracted_text = text
				except Exception as e:
					st.error(f"Failed to extract content: {e}")
			extracted_text_msg = st.empty()
			extracted_text_msg.success("Content extracted successfully!")
			time.sleep(3)
			extracted_text_msg.empty()

		# Show sample document preview
		if st.session_state.extracted_text:
			st.text("Sample Preview:")
			st.text_area(label="", value=st.session_state.extracted_text[:1000], height=150)

		# Chunk and store in Pinecone DB
		if st.session_state.extracted_text and not st.session_state.chunks_stored:
			with st.spinner("Reading and understanding your content..."):
				try:
					self.chunker.chunk_and_store(st.session_state.extracted_text)
					st.session_state.chunks_stored = True
				except Exception as e:
					st.error(f"Failed to chunk and store: {e}")
			chunks_stored_msg = st.empty()
			chunks_stored_msg.success("Done!")
			time.sleep(3)
			chunks_stored_msg.empty()

		# Get user query
		if st.session_state.chunks_stored:
			st.markdown("---")
			st.subheader("Ask a question 🤔")
			user_query = st.text_input("")
			st.session_state.user_query = user_query

		# Retrieve relevant information
		if st.session_state.user_query:
			with st.spinner("Searching and generating response..."):
				try:
					answer = self.retriever.retrieve(user_query)
					st.session_state.answer = answer
				except Exception as e:
					st.error(f"Failed to search user query: {e}")

		# Generate answer
		if st.session_state.answer:
			st.subheader("Answer 🎯")
			st.text(st.session_state.answer)

if __name__ == "__main__":
	app = RAGApp()
	app.main()