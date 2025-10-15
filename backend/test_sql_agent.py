# backend/test_sql_agent.py
import pytest
from backend.sql_agent import build_graph, process_query
from langchain_community.utilities import SQLDatabase

@pytest.fixture
def db():
    return SQLDatabase.from_uri("sqlite:///custom_database.db")

def test_process_query(db):
    graph, _ = build_graph(db)
    result = process_query(graph, "SELECT * FROM test_table LIMIT 1")
    assert "answer" in result
    assert result["viz_type"] in ["bar", "horizontal_bar", "line", "pie", "scatter", "none"]
    assert isinstance(result["viz_data"], dict)

def test_empty_result(db):
    graph, _ = build_graph(db)
    result = process_query(graph, "SELECT * FROM nonexistent_table")
    assert "error" in result["answer"].lower()
    assert result["viz_type"] == "none"