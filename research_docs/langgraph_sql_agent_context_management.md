# LangGraph and SQL Agent Context Management

## Overview

This document provides authoritative information about LangGraph and SQL agents, with a focus on context management and retention strategies. It covers best practices for maintaining conversation context, managing state, and implementing effective memory patterns in LangGraph-based applications.

## LangGraph Fundamentals

### What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, based on the Pregel model. It extends LangChain by providing:

1. **Explicit State Management**: Clear definition of application state
2. **Persistent Memory**: Built-in checkpointing for conversation history
3. **Cycles and Branching**: Support for loops and conditional logic
4. **Human-in-the-Loop**: Interruptions and approvals during execution

### Core Components

1. **StateGraph**: The main class for defining workflows
2. **Nodes**: Functions that transform the graph's state
3. **Edges**: Conditional or static transitions between nodes
4. **Checkpointer**: Handles persistence of state across executions
5. **Store**: Long-term memory for application-specific data

## Memory and Context Management in LangGraph

### Short-term Memory (Thread-level Persistence)

LangGraph provides built-in mechanisms for maintaining conversation context through checkpointers:

#### InMemorySaver (Development)
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

#### Production Checkpointers
For production environments, use database-backed checkpointers:

##### PostgreSQL
```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

##### MongoDB
```python
from langgraph.checkpoint.mongodb import MongoDBSaver

DB_URI = "mongodb://localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

##### Redis
```python
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### Thread Identification

Each conversation thread is identified by a `thread_id` in the configuration:

```python
config = {"configurable": {"thread_id": "unique_conversation_id"}}
response = graph.invoke(input_data, config)
```

### Accessing and Managing Checkpoints

#### View Current State
```python
current_state = graph.get_state(config)
```

#### View State History
```python
history = list(graph.get_state_history(config))
```

#### Checkpointer API Direct Access
```python
checkpoint_tuple = checkpointer.get_tuple(config)
all_checkpoints = list(checkpointer.list(config))
```

## SQL Agent Context Management Best Practices

### State Schema Design

Design your state schema to effectively capture all necessary context:

```python
from typing import Annotated, List, Dict
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    """State definition for the LangGraph workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    sql_result: List[Dict]
    viz_type: str
    viz_data: Dict
    tables: List[Dict]
    tool_history: List[Dict]  # Store tool call inputs/outputs
    status_messages: List[str]  # Store node processing status
```

### Message History Management

#### Automatic Message Accumulation
Using `add_messages` reducer automatically accumulates messages:

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

#### Manual Message Management
For more control, implement custom reducers:

```python
def custom_message_reducer(current: List[AnyMessage], update: List[AnyMessage]) -> List[AnyMessage]:
    # Custom logic for message handling
    combined = current + update
    # Apply limits, filtering, or other logic
    return combined[-20:]  # Keep only last 20 messages
```

### Context Summarization

For long conversations, implement summarization to prevent context window overflow:

```python
from langmem.short_term import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

# Add to graph
builder.add_node("summarize", summarization_node)
```

### Custom Context Management

Implement custom context management for domain-specific needs:

```python
def manage_context(state: State) -> Dict:
    """Custom context management logic."""
    # Extract relevant context from messages
    conversation_history = state.get("messages", [])
    
    # Maintain tool call history
    tool_history = state.get("tool_history", [])
    
    # Track conversation flow
    status_messages = state.get("status_messages", [])
    
    # Return updated context
    return {
        "relevant_history": conversation_history[-10:],  # Last 10 messages
        "recent_tools": tool_history[-5:],  # Last 5 tool calls
        "current_status": status_messages[-3:]  # Last 3 status updates
    }
```

## Best Practices for Context Retention

### 1. Efficient State Design

- Only store necessary information in state
- Use appropriate data structures for quick access
- Consider serialization costs for large data

### 2. Memory Optimization

#### Token-based Trimming
```python
def trim_messages_by_tokens(messages: List[AnyMessage], max_tokens: int = 4000) -> List[AnyMessage]:
    """Trim messages to stay within token limits."""
    total_tokens = 0
    trimmed_messages = []
    
    # Process messages in reverse order (most recent first)
    for message in reversed(messages):
        message_tokens = count_tokens_approximately([message])
        if total_tokens + message_tokens <= max_tokens:
            trimmed_messages.insert(0, message)
            total_tokens += message_tokens
        else:
            break
    
    return trimmed_messages
```

#### Semantic Chunking
```python
def semantic_chunking(messages: List[AnyMessage], chunk_size: int = 5) -> List[List[AnyMessage]]:
    """Group messages semantically."""
    chunks = []
    current_chunk = []
    
    for message in messages:
        current_chunk.append(message)
        # Create new chunk based on semantic boundaries or size
        if len(current_chunk) >= chunk_size or is_semantic_boundary(message):
            chunks.append(current_chunk)
            current_chunk = []
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
```

### 3. Checkpointing Strategy

#### Frequency Control
```python
# Configure checkpointer with appropriate settings
checkpointer = PostgresSaver.from_conn_string(DB_URI)
# Set checkpoint frequency based on application needs
```

#### Selective Checkpointing
```python
# Only checkpoint at critical points in the workflow
def should_checkpoint(state: State) -> bool:
    """Determine if current state should be checkpointed."""
    # Custom logic based on state content or workflow position
    return len(state.get("messages", [])) % 5 == 0  # Every 5 messages
```

### 4. Long-term Memory Integration

For persistent user or application data, use LangGraph Stores:

```python
from langgraph.store.memory import InMemoryStore

# Development
store = InMemoryStore()

# Production
from langgraph.store.postgres import PostgresStore
store = PostgresStore.from_conn_string(DB_URI)

graph = builder.compile(store=store, checkpointer=checkpointer)
```

### 5. Context-aware Tool Design

Make tools context-aware by injecting state:

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

def context_aware_tool(
    query: str,
    state: Annotated[State, InjectedState]
) -> str:
    """Tool that uses conversation context."""
    # Access previous messages
    history = state.get("messages", [])
    
    # Access previous tool calls
    tool_history = state.get("tool_history", [])
    
    # Use context to improve tool response
    # ...
    
    return result
```

## Implementation Patterns

### Pattern 1: Message History with Limits

```python
from langchain_core.messages import RemoveMessage

def limit_message_history(state: State) -> Dict:
    """Keep only the most recent messages."""
    messages = state.get("messages", [])
    if len(messages) > 20:  # Limit to 20 messages
        # Remove oldest messages
        remove_messages = [RemoveMessage(id=m.id) for m in messages[:-20]]
        return {"messages": remove_messages}
    return {}
```

### Pattern 2: Context Summarization Node

```python
def summarize_context(state: State) -> Dict:
    """Generate summary of conversation context."""
    messages = state.get("messages", [])
    
    if len(messages) > 10:  # Summarize after 10 messages
        summary_prompt = f"""
        Summarize the following conversation:
        {format_messages(messages[-10:])}
        
        Summary:
        """
        
        summary = model.invoke(summary_prompt)
        
        # Replace old messages with summary
        remove_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
        return {
            "messages": remove_messages,
            "context_summary": summary.content
        }
    
    return {}
```

### Pattern 3: Adaptive Context Management

```python
def adaptive_context_manager(state: State) -> Dict:
    """Adaptively manage context based on conversation complexity."""
    messages = state.get("messages", [])
    current_message = messages[-1] if messages else None
    
    actions = []
    
    # Check if we need to summarize
    if len(messages) > 15:
        actions.append("summarize")
    
    # Check if we need to trim
    if len(messages) > 25:
        actions.append("trim")
    
    # Check for topic shift
    if detects_topic_shift(messages):
        actions.append("reset_context")
    
    return {"context_actions": actions}
```

## Monitoring and Debugging

### State Inspection

```python
# View current state
state = graph.get_state(config)
print(f"Current messages: {len(state.values['messages'])}")

# View state history
history = list(graph.get_state_history(config))
for snapshot in history:
    print(f"Step {snapshot.metadata['step']}: {len(snapshot.values['messages'])} messages")
```

### Performance Metrics

Track context management performance:

```python
import time

def track_context_performance(func):
    """Decorator to track context management performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log performance metrics
        logger.info(f"Context operation took {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper
```

## Conclusion

Effective context management in LangGraph-based SQL agents requires:

1. **Proper State Design**: Thoughtful state schema that captures all necessary context
2. **Appropriate Checkpointing**: Using the right checkpointer for development vs. production
3. **Memory Optimization**: Implementing trimming and summarization strategies
4. **Context-aware Tools**: Making tools sensitive to conversation history
5. **Monitoring**: Tracking context management performance and effectiveness

By following these best practices, you can build robust SQL agents that maintain coherent conversation context while efficiently managing memory resources.
