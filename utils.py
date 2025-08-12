import os
import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Utils:
    def __init__(self):
        self.llm_model = "gpt-4o-mini"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._init_llm()
        self._init_mlflow()

    def _init_llm(self):
        """ Initialize LLM. """
        self.llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_key=self.api_key,
            temperature=0.2
        )
    
    def _init_mlflow(self):
        """ Initialize MLflow tracking. """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI")) # Connect to remote MLflow server
        mlflow.set_experiment("Agentic RAG")
        mlflow.dspy.autolog(log_evals=True, log_compiles=True, log_traces_from_compile=True) # Log and Track DSPy