# Agentic RAG
An Advanced Multi-hop Agentic RAG system with:
* Multi-format document Ingestion.
* Query Expansion with Multi-query Augmentation.
* Cross-Encoder Re-ranking.
* Intelligent ReAct Agent for reasoning.
* Web search Tool Calling.
* End-to-End Tracking.

[![Demo Video](./images/thumbnail.png)](https://drive.google.com/file/d/1EiXrJsbNTXtrkE-GgrNWxp80FYGnN2ue/view?usp=drive_link)

### System Architecture:
![workflow](./images/RAG_APP_Workflow.drawio.png)


### Tools:
* ![langchain](./images/langchain.png) ➡️ Text Chunking
* ![embedding](./images/openai_embedding.png) ➡️ Vector Embedding
* ![pinecone](./images/pinecone.png) ➡️ Vector Storage Database, Re-ranking
* ![dspy](./images/dspy_logo.png) ➡️ Chain Of Thought & ReAct Agent
* ![chatgpt](./images/chatgpt.jpg) ➡️ LLM & Query Expansion
* ![tavily](./images/tavily.png) ➡️ Web Search Tool
* ![mlflow](./images/MLflow.png) ➡️ End-to-End Tracking
* ![streamlit](./images/streamlit.png) ➡️ User Interface


### Example:
* Upload ```microsoft_annual_report_2024.pdf``` document.
* Ask: "*How much has the revenue increased since 2023? and what is Google's annual revenue?*"
* The agent splits the complex query into two smaller queries and addresses both.
* The ReAct agent uses Chain of Thought reasoning to answer the first query - reasons that the answer is found in the retrieved content.
![mlflow_1](./images/mlflow_1.png)
* The ReAct agent decides to use the Web Search Tool since the answer to the second query is not found in the retrieved content.
![mlflow_2](./images/mlflow_2.png)


### Instructions to run code:
* To start application: ```python -m streamlit run main.py```
* To start mlflow server: ```mlflow server --host 127.0.0.1 --port 8080```
