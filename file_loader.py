import os
import pdfplumber
from docx import Document
from newspaper import Article
from urllib.parse import urlparse

class FileLoader:
	"""
	Class to handle files (pdf, docx, txt) and web articles.
	"""
	def _is_url(self, string):
		""" Checks if the uploaded one is a URL. """
		try:
			result = urlparse(string)
			return all([result.scheme in ("http", "https"), result.netloc])
		except:
			return False

	def _load_url(self, url):
		""" Loads the URL and extract the text. """
		article = Article(url)
		article.download()
		article.parse()
		return article.text.strip()

	
	def _load_pdf(self, filepath):
		""" Loads a PDF and extract the text. """
		with pdfplumber.open(filepath) as pdf:
			return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

	def _load_docx(self, filepath):
		""" Loads a docx and extract the text. """
		doc = Document(filepath)
		return "\n".join(p.text for p in doc.paragraphs)

	def _load_txt(self, filepath):
		""" Loads a text file and extract the text. """
		with open(filepath, "r", encoding="utf-8") as f:
			return f.read()

	def load(self, filepath_or_url):
		""" Loads and extracts content. """
		if self._is_url(filepath_or_url):
			return self._load_url(filepath_or_url)
		else:
			ext = os.path.splitext(filepath_or_url)[1].lower()

			if ext == ".pdf":
				return self._load_pdf(filepath_or_url)
			elif ext == ".docx":
				return self._load_docx(filepath_or_url)
			elif ext == ".txt":
				return self._load_txt(filepath_or_url)
			else:
				raise ValueError(f"Unsupported file format: {ext}")