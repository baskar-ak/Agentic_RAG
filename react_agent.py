import os
import dspy
import mlflow
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

class ReActAgent(dspy.Signature):
    """Use the context to answer the question. If not, use a web search."""
    context: str = dspy.InputField(desc="Context from the knowledge base.")
    query: str = dspy.InputField(desc="User query.")
    output: str = dspy.OutputField(desc="Answer to user's query.")


class WebSearch:
    """ 
    Websearch tool using Tavily.
    """
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def web_search_tool(self, user_query):
        """ Websearch Tool. """
        result = self.tavily_client.search(query=user_query, max_results=3)
        context = "\n".join([item["content"] for item in result["results"]])
        return context


class Reacter:
    """
    Reacter Agent. Uses Tool Calling (websearch tool), if provided content is not sufficient to answer user query.
    """
    def __init__(self):
        self.web_search = WebSearch()

    def generate_answer(self, context, query):
        """ Generates answer for user query. """
        with mlflow.start_run(run_name="ReAct Agent", nested=True):
            mlflow.log_param("user_query", query)
            mlflow.log_text(context, "context.txt")

            react_agent = dspy.ReAct(ReActAgent, tools=[self.web_search.web_search_tool]) # Reasoning + Action
            answer = react_agent(context=context, query=query)

            mlflow.log_text(answer["reasoning"], "react_reasoning.txt")
            mlflow.log_text(answer["output"], "react_output.txt")

            return answer["output"]
