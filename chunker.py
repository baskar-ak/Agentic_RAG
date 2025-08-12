import os
import mlflow
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class Chunker:
	"""
	Class that chunks the content, embeds and stores it in Pinecone DB.
	"""
	def __init__(self):
		self._init_text_splitter()
		self._init_embeddings()
		self._init_pinecone()

	def _init_text_splitter(self, chunk_size=500, chunk_overlap=100):
		""" Initialize text splitter for text chunking. """
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.text_splitter = RecursiveCharacterTextSplitter(
				separators = ["\n\n", "\n", ". ", " ", ""],
				chunk_size = self.chunk_size,
				chunk_overlap = self.chunk_overlap
			)

	def _init_embeddings(self):
		""" Initialize text embedder. """
		self.embeddings = OpenAIEmbeddings(
			model="text-embedding-3-small",
			openai_api_key=os.getenv("OPENAI_API_KEY")
		)

	def _init_pinecone(self):
		""" Initialize Pinecone DB. """
		self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
		self.pc_index_name = "langchain-openai-pinecone-rag"

		if not self.pc.has_index(self.pc_index_name):
			self.pc.create_index(
					name = self.pc_index_name,
					dimension = 1536, # Dimension of embedding produced by OpenAI's text-embedding-3-small
					metric = "cosine", # Search metric. Other metrics include "Euclidean, Dotproduct"
					spec = ServerlessSpec(
							cloud = "aws",
							region = "us-east-1"
						)
				)

		self.pc_index_client = self.pc.Index(self.pc_index_name) # Initialize index client

	def chunk_and_store(self, raw_text):
		""" Chunks input text, embeds each chunk and stores in Pinecone. Returns the number of chunks stored. """
		chunks = self.text_splitter.split_text(raw_text) # Split text to chunks
		chunk_embeddings = self.embeddings.embed_documents(chunks) # Turn chunks into embeddings
		ids = [str(i) for i in range(len(chunks))] # IDs for chunks

		embedding_vectors = [
		{
			"id" : ids[i],
			"values" : chunk_embeddings[i],
			"metadata" : {
				"text" : chunks[i],
			}
		}
		for i in range(len(chunks))
		]

		self.pc_index_client.upsert(embedding_vectors) # Insert and update db
		with mlflow.start_run(run_name="RAG Chunker"):
			mlflow.log_metric("num_chunks_stored_pinecone", len(chunks)) # Logs
			mlflow.log_metric("chunk_length", self.chunk_size)