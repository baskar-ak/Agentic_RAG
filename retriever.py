import os
import dspy
import mlflow
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, RerankModel
from utils import Utils
from react_agent import Reacter

load_dotenv()

class QuerySplitter(dspy.Signature):
    """Split the query only if it has multiple parts to it."""
    query: str = dspy.InputField()
    output: list[str] = dspy.OutputField(desc="Query or subqueries.")

class RetrieverAgent:
    def __init__(self):
        self.pc_index_name = "langchain-openai-pinecone-rag"
        self.embed_model = "text-embedding-3-small"
        self.namespace = "__default__"
        self.utils = Utils()
        self.reacter = Reacter()
        self._init_embeddings()
        self._init_pinecone()

    def _init_embeddings(self):
        self.embeddings = OpenAIEmbeddings(
            model=self.embed_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def _init_pinecone(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Initialize Pinecone
        self.pc_index_client = self.pc.Index(self.pc_index_name)  # Initialize index client
    
    def split_query(self, user_query):
        """ Splits the user query if it has multiple parts. """
        self.query_splitter = dspy.ChainOfThought(QuerySplitter) # Chain Of Thought dspy module
        subqueries = self.query_splitter(query=user_query)
        return subqueries["output"]

    def augment_query(self, user_query, n_variants=3):
        """ Multiple Query Expansion. """
        prompt = (
            f"""Given the question: "{user_query}", generate {n_variants} related questions with alternative phrasings. 
            Provide only short and complete questions without compound sentences. 
            Output one question per line and do not number the questions."""
        )
        response = self.utils.llm.invoke(prompt)
        variants = [line.strip("- ").strip() for line in response.content.split("\n") if line.strip()]
        return [user_query] + variants

    def query_pinecone(self, user_query, top_k=10):
        """ Expand query, embed and retrieve from pinecone. """
        expanded_queries = self.augment_query(user_query)  # Augment queries
        joint_query_embedding = self.embeddings.embed_documents(expanded_queries) # Join the queries and embed them
        avg_embedding = [sum(values) / len(values) for values in zip(*joint_query_embedding)]  # Average the embedding vectors

        results = self.pc_index_client.query( # Query pinecone and retrieve 10 relevant documents
            vector=avg_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        return results

    def rerank_results(self, user_query, results):
        """ Cross-encoder Re-ranking. """
        matches = results.get("matches", [])
        documents = [match["metadata"]["text"] for match in matches]

        reranked = self.pc.inference.rerank( # Rerank results from pinecone search results
            model=RerankModel.Bge_Reranker_V2_M3,
            query=user_query,
            documents=documents,
            top_n=5,  # Get top 5
            return_documents=True,
        )
        return reranked

    def retrieve(self, user_query):
        """ Retrieval pipeline with MLflow tracking. """
        with mlflow.start_run(run_name="RAG Retrieve"):
            mlflow.log_param("user_query", user_query)

            full_answer = ""
            subqueries = self.split_query(user_query) # Get subqueries if any

            mlflow.log_metric("num_subqueries", len(subqueries))

            for i, subquery in enumerate(subqueries): # For each subquery do the following
                mlflow.log_param(f"subquery_{i}", subquery) # Log each subquery
                search_results = self.query_pinecone(subquery) # Get search results by quering pinecone
                reranked_results = self.rerank_results(subquery, search_results) # Re-rank the search results
                documents_text = [item["document"]["text"] for item in reranked_results.get("data", [])] # Extract the content
                mlflow.log_metric(f"num_docs_{i}", len(documents_text)) # Log number of retrieved documents
                subcontext = "\n".join(documents_text)
                mlflow.log_text(subcontext, artifact_file=f"context_{i}.txt")
                full_answer = full_answer + " " + self.reacter.generate_answer(subcontext, subquery) # Generate the answer
            
            mlflow.log_text(full_answer.strip(), "final_answer.txt")
            return full_answer.strip()
