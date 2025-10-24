# SQL Agents: Patterns and Context Management

## Introduction

SQL agents are specialized AI systems designed to interact with databases through natural language queries. They translate user requests into SQL commands, execute them against databases, and present results in an understandable format. Effective context management is crucial for SQL agents to maintain coherent conversations and provide accurate responses across multiple interactions.

## SQL Agent Architecture Patterns

### 1. Basic SQL Agent Pattern

The simplest SQL agent follows a linear workflow:
1. Receive natural language query
2. Translate to SQL
3. Execute SQL against database
4. Format and return results

```python
# Basic pattern
def basic_sql_agent(question: str, db: SQLDatabase) -> str:
    # 1. Generate SQL from natural language
    sql_query = generate_sql(question)
    
    # 2. Execute query
    result = db.run(sql_query)
    
    # 3. Format result
    formatted_result = format_result(result, question)
    
    return formatted_result
```

### 2. Tool-Augmented SQL Agent Pattern

More sophisticated agents use tools for specialized functions:
1. Question analysis
2. Schema inspection
3. Query validation
4. Result verification

```python
# Tool-augmented pattern
class SQLAgent:
    def __init__(self, db: SQLDatabase, llm: BaseLanguageModel):
        self.db = db
        self.llm = llm
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        return toolkit.get_tools()
    
    def process_query(self, question: str, context: List[Message]) -> str:
        # Use tools in sequence
        tables = self._list_tables()
        schema = self._get_schema(tables)
        query = self._generate_query(question, schema, context)
        validated_query = self._validate_query(query)
        result = self._execute_query(validated_query)
        verified_result = self._verify_result(result, question)
        
        return self._format_response(verified_result, question)
```

### 3. LangGraph-Based SQL Agent Pattern

The most advanced pattern uses LangGraph for state management:
1. Explicit state definition
2. Persistent memory
3. Complex workflows with branching
4. Human-in-the-loop capabilities

```python
# LangGraph pattern
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    sql_result: List[Dict]
    # ... other state fields

def agent_node(state: State, tools) -> Dict:
    # Agent logic with tool binding
    pass

def tool_node(state: State, tools) -> Dict:
    # Tool execution logic
    pass

# Graph construction
workflow = StateGraph(State)
workflow.add_node("agent", lambda state: agent_node(state, tools))
workflow.add_node("tools", lambda state: tool_node(state, tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# Compile with checkpointer for persistence
checkpointer = PostgresSaver.from_conn_string(DB_URI)
graph = workflow.compile(checkpointer=checkpointer)
```

## Context Management Strategies

### 1. Conversation History Management

Maintaining conversation context is essential for coherent interactions:

#### Message Accumulation
```python
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

#### History Filtering
```python
def filter_relevant_history(messages: List[AnyMessage], current_question: str) -> List[AnyMessage]:
    """Filter conversation history to relevant messages only."""
    relevant_messages = []
    
    # Always include recent messages
    recent_messages = messages[-5:]  # Last 5 messages
    
    # Include messages that mention relevant tables/columns
    mentioned_entities = extract_entities(current_question)
    entity_related_messages = [
        msg for msg in messages 
        if any(entity in msg.content for entity in mentioned_entities)
    ]
    
    # Combine and deduplicate
    all_relevant = set(recent_messages + entity_related_messages)
    return sorted(all_relevant, key=lambda x: messages.index(x))
```

### 2. Schema and Metadata Context

Preserve database schema information to avoid redundant tool calls:

```python
class DatabaseContext:
    def __init__(self):
        self.table_schemas = {}
        self.column_descriptions = {}
        self.sample_data = {}
        self.query_history = []
    
    def update_schema(self, table_name: str, schema_info: Dict):
        """Update schema information for a table."""
        self.table_schemas[table_name] = schema_info
    
    def get_relevant_schema(self, question: str) -> Dict:
        """Get schema information relevant to the current question."""
        relevant_tables = identify_tables_in_question(question)
        return {
            table: self.table_schemas.get(table, {})
            for table in relevant_tables
        }
```

### 3. Query Result Context

Maintain context about previous query results:

```python
class QueryResultContext:
    def __init__(self):
        self.results_cache = {}
        self.result_summaries = {}
    
    def cache_result(self, query: str, result: List[Dict]):
        """Cache query result with timestamp."""
        self.results_cache[hash(query)] = {
            'result': result,
            'timestamp': datetime.now(),
            'summary': self._summarize_result(result)
        }
    
    def get_cached_result(self, query: str, max_age_minutes: int = 30) -> Optional[List[Dict]]:
        """Retrieve cached result if still valid."""
        cache_key = hash(query)
        if cache_key in self.results_cache:
            cached = self.results_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(minutes=max_age_minutes):
                return cached['result']
        return None
```

## Advanced Context Management Techniques

### 1. Adaptive Context Windows

Dynamically adjust the amount of context based on complexity:

```python
def calculate_context_window(question: str, conversation_length: int) -> int:
    """Calculate optimal context window size."""
    base_window = 10
    
    # Increase for complex questions
    if is_complex_query(question):
        base_window += 5
    
    # Decrease for simple follow-ups
    if is_simple_followup(question):
        base_window -= 3
    
    # Cap based on conversation length
    return min(base_window, conversation_length, 20)
```

### 2. Context Summarization

Summarize lengthy conversations to stay within token limits:

```python
def summarize_conversation_context(messages: List[AnyMessage]) -> str:
    """Create a summary of conversation context."""
    if len(messages) <= 10:
        return None  # No need to summarize short conversations
    
    summary_prompt = f"""
    Summarize the following conversation for use as context in future queries:
    
    CONVERSATION:
    {format_messages(messages[-15:])}  # Last 15 messages
    
    SUMMARY GUIDELINES:
    1. Key facts established in the conversation
    2. Tables and columns discussed
    3. Previous query results that might be relevant
    4. User preferences or constraints mentioned
    
    SUMMARY:
    """
    
    summary = llm.invoke(summary_prompt)
    return summary.content
```

### 3. Entity Resolution and Coreference

Resolve references to previously mentioned entities:

```python
def resolve_entity_references(question: str, context: List[AnyMessage]) -> str:
    """Resolve pronouns and references to previous entities."""
    # Extract entities from context
    context_entities = extract_all_entities(context)
    
    # Resolve references in current question
    resolved_question = question
    for reference, entity in resolve_references(question, context_entities):
        resolved_question = resolved_question.replace(reference, entity)
    
    return resolved_question
```

## Memory Patterns for SQL Agents

### 1. Short-term Memory (Conversation Context)

Managed through LangGraph checkpointers:

```python
# Short-term memory via checkpointer
config = {"configurable": {"thread_id": session_id}}
response = graph.invoke({"question": user_input}, config)

# Retrieve conversation history
history = list(graph.get_state_history(config))
```

### 2. Medium-term Memory (Session Context)

Stored in application state or temporary storage:

```python
class SessionContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.preferences = {}
        self.recent_queries = []
        self.frequently_used_tables = []
    
    def update_preferences(self, preference: Dict):
        """Update user preferences for this session."""
        self.preferences.update(preference)
    
    def add_query(self, query: str, result_summary: str):
        """Add query to recent history."""
        self.recent_queries.append({
            'query': query,
            'result': result_summary,
            'timestamp': datetime.now()
        })
        # Keep only last 20 queries
        self.recent_queries = self.recent_queries[-20:]
```

### 3. Long-term Memory (User/Application Context)

Persisted across sessions:

```python
# Long-term memory via LangGraph Store
class UserPreferencesStore:
    def __init__(self, store: BaseStore):
        self.store = store
        self.namespace = ("user_preferences",)
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Retrieve user preferences."""
        preferences = self.store.search((*self.namespace, user_id))
        return preferences[0].value if preferences else {}
    
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences."""
        self.store.put((*self.namespace, user_id), str(uuid.uuid4()), preferences)
```

## Error Handling and Context Recovery

### 1. Context-Aware Error Recovery

Use conversation context to recover from errors:

```python
def recover_from_query_error(error: Exception, context: State) -> str:
    """Attempt to recover from SQL query errors using context."""
    if "table not found" in str(error):
        # Suggest alternative tables based on context
        suggested_tables = find_similar_tables_in_context(
            extract_table_name_from_error(error),
            context.get("messages", [])
        )
        return f"Did you mean one of these tables: {', '.join(suggested_tables)}?"
    
    elif "column not found" in str(error):
        # Suggest alternative columns
        suggested_columns = find_similar_columns_in_context(
            extract_column_name_from_error(error),
            context.get("schema_info", {})
        )
        return f"Did you mean one of these columns: {', '.join(suggested_columns)}?"
    
    else:
        return f"I encountered an error: {str(error)}. Could you rephrase your question?"
```

### 2. Context Validation

Validate that context is still relevant:

```python
def validate_context_relevance(context: State, current_question: str) -> bool:
    """Validate that conversation context is still relevant."""
    # Check if tables mentioned in question have changed
    question_tables = extract_tables(current_question)
    context_tables = extract_tables_from_context(context.get("messages", []))
    
    # If no overlap and context is old, context may be stale
    if not set(question_tables) & set(context_tables):
        last_message_time = get_last_message_time(context.get("messages", []))
        if datetime.now() - last_message_time > timedelta(hours=1):
            return False
    
    return True
```

## Performance Optimization

### 1. Context Caching

Cache frequently accessed context elements:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_table_schema_cached(table_name: str, db_connection_string: str) -> Dict:
    """Cached retrieval of table schema."""
    # Implementation that caches schema information
    pass
```

### 2. Lazy Context Loading

Load context only when needed:

```python
class LazyContextLoader:
    def __init__(self, context_provider):
        self.context_provider = context_provider
        self._cached_context = None
    
    @property
    def context(self):
        """Lazy load context only when accessed."""
        if self._cached_context is None:
            self._cached_context = self.context_provider.load_context()
        return self._cached_context
    
    def invalidate_cache(self):
        """Invalidate cached context."""
        self._cached_context = None
```

## Monitoring and Analytics

### 1. Context Usage Metrics

Track how context is being used:

```python
class ContextMetrics:
    def __init__(self):
        self.metrics = {
            'context_length_avg': 0,
            'context_truncations': 0,
            'context_cache_hits': 0,
            'context_cache_misses': 0
        }
    
    def record_context_usage(self, context_length: int, was_truncated: bool, cache_hit: bool):
        """Record context usage metrics."""
        self.metrics['context_length_avg'] = (
            self.metrics['context_length_avg'] * 0.9 + context_length * 0.1
        )
        
        if was_truncated:
            self.metrics['context_truncations'] += 1
            
        if cache_hit:
            self.metrics['context_cache_hits'] += 1
        else:
            self.metrics['context_cache_misses'] += 1
```

### 2. Context Quality Assessment

Evaluate the quality of context management:

```python
def assess_context_quality(conversation: List[AnyMessage]) -> Dict:
    """Assess the quality of context management in a conversation."""
    metrics = {
        'coherence_score': calculate_coherence(conversation),
        'redundancy_score': calculate_redundancy(conversation),
        'relevance_score': calculate_relevance(conversation),
        'efficiency_score': calculate_efficiency(conversation)
    }
    
    return metrics
```

## Best Practices Summary

### 1. Design Principles

- **Explicit State Management**: Clearly define what context to maintain
- **Progressive Disclosure**: Start with minimal context and add as needed
- **Context Freshness**: Regularly validate that context is still relevant
- **Performance Awareness**: Monitor context management overhead

### 2. Implementation Guidelines

- Use LangGraph's built-in checkpointing for conversation persistence
- Implement context summarization for long conversations
- Cache frequently accessed context elements
- Validate context relevance before use
- Provide graceful degradation when context is unavailable

### 3. Monitoring Recommendations

- Track context length and truncation frequency
- Monitor cache hit rates for context elements
- Measure response quality correlation with context management
- Log context-related errors for debugging

## Conclusion

Effective context management is fundamental to building capable SQL agents. By implementing appropriate patterns for conversation history, schema awareness, result caching, and error recovery, SQL agents can maintain coherent interactions while efficiently managing computational resources. The LangGraph framework provides powerful primitives for implementing these patterns, enabling developers to build sophisticated context-aware SQL agents.
