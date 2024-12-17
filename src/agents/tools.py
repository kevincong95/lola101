import math
import os
import re
from dotenv import load_dotenv

from langchain_core.tools import BaseTool, tool
from neo4j import GraphDatabase, Record

# Neo4j connection setup
load_dotenv()
password = os.getenv('NEO4J_PASSWORD')
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))


def query_neo4j_func(questionId: str) -> Record:
    """Execute a Neo4j query and return the result."""
    query = """
        MATCH (q:Question) WHERE (q.LolQuestionIndex.startsWith({questionId: $questionId}))
        RETURN q.QuestionTitle, q.QuestionDescription, q.GoldenSolution
        ORDER BY rand()
        LIMIT 1
        """
    result, _, _ = driver.execute_query(query, {'questionId': questionId})
    record = result.single()
    return record

query_neo4j: BaseTool = tool(query_neo4j_func)
query_neo4j.name = "Query Neo4j"
