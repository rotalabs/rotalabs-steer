# Tutorial: LangChain Integration

Use steering vectors with LangChain chat models and agents.

## Installation

```bash
pip install rotalabs-steer[langchain]
```

## SteeredChatModel

The primary integration point for chat-based applications.

### Basic Usage

```python
from rotalabs_steer.integrations.langchain import SteeredChatModel
from langchain_core.messages import HumanMessage, SystemMessage

chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    steering_configs={
        "refusal": {
            "vector_path": "./refusal_vectors/layer_15",
            "strength": 1.0,
        },
    },
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello! How are you?"),
]

response = chat.invoke(messages)
print(response.content)
```

### Multiple Behaviors

```python
chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    steering_configs={
        "refusal": {
            "vector_path": "./vectors/refusal_layer_15",
            "strength": 1.0,
        },
        "uncertainty": {
            "vector_path": "./vectors/uncertainty_layer_14",
            "strength": 0.5,
        },
        "tool_restraint": {
            "vector_path": "./vectors/tool_restraint_layer_16",
            "strength": 0.8,
        },
    },
)
```

### Dynamic Strength Adjustment

```python
# Start with high refusal
response1 = chat.invoke([HumanMessage(content="How do I pick a lock?")])

# Reduce for legitimate questions
chat.set_strength("refusal", 0.3)
response2 = chat.invoke([HumanMessage(content="How do locksmiths work?")])

# Temporarily disable
chat.disable_steering("refusal")
response3 = chat.invoke([HumanMessage(content="Explain lock mechanisms")])

# Re-enable
chat.enable_steering("refusal", 1.0)
```

### Adding Vectors at Runtime

```python
# Start with no steering
chat = SteeredChatModel(model_name="Qwen/Qwen3-8B")

# Add vector later
chat.add_vector(
    behavior="refusal",
    vector="./vectors/refusal_layer_15",
    strength=1.0,
)
```

## Using with LangChain Chains

### Simple Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant."),
    ("human", "{question}"),
])

chain = prompt | chat | StrOutputParser()

answer = chain.invoke({"question": "Write a hello world in Python"})
```

### With Memory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
)

# Conversation with memory
response1 = chain_with_history.invoke(
    {"question": "My name is Alice"},
    config={"configurable": {"session_id": "abc123"}},
)

response2 = chain_with_history.invoke(
    {"question": "What's my name?"},
    config={"configurable": {"session_id": "abc123"}},
)
```

## SteeredAgentExecutor

For tool-using agents with steering control.

### Basic Agent

```python
from rotalabs_steer.integrations.langchain import SteeredAgentExecutor
from langchain.tools import Tool

# Define tools
def search(query: str) -> str:
    return f"Search results for: {query}"

def calculator(expr: str) -> str:
    return str(eval(expr))

tools = [
    Tool(name="search", func=search, description="Search the web"),
    Tool(name="calculator", func=calculator, description="Calculate math"),
]

# Create agent with tool restraint
agent = SteeredAgentExecutor(
    model_name="Qwen/Qwen3-8B",
    tools=tools,
    steering_configs={
        "tool_restraint": {
            "vector_path": "./vectors/tool_restraint_layer_16",
            "strength": 1.0,
        },
    },
    system_prompt="You are a helpful assistant. Use tools only when necessary.",
)

# Run
result = agent.invoke("What is 2 + 2?")
print(result["output"])
```

### Combining Multiple Behaviors

```python
agent = SteeredAgentExecutor(
    model_name="Qwen/Qwen3-8B",
    tools=tools,
    steering_configs={
        "refusal": {
            "vector_path": "./vectors/refusal_layer_15",
            "strength": 1.0,
        },
        "tool_restraint": {
            "vector_path": "./vectors/tool_restraint_layer_16",
            "strength": 0.8,
        },
        "uncertainty": {
            "vector_path": "./vectors/uncertainty_layer_14",
            "strength": 0.5,
        },
    },
)
```

## Configuration Options

### Device Selection

```python
chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    device="auto",  # Automatically select best device
    # device="cuda",    # Force CUDA
    # device="mps",     # Apple Silicon
    # device="cpu",     # CPU only
)
```

### Dtype Selection

```python
chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    torch_dtype="float16",   # Default, good balance
    # torch_dtype="bfloat16",  # Better for some models
    # torch_dtype="float32",   # Full precision
)
```

### Generation Parameters

```python
chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

# Override per-call
response = chat.invoke(
    messages,
    max_new_tokens=512,
    temperature=0.9,
)
```

## Error Handling

```python
from rotalabs_steer.integrations import HAS_LANGCHAIN

if not HAS_LANGCHAIN:
    raise ImportError(
        "LangChain not installed. Run: pip install rotalabs-steer[langchain]"
    )

try:
    chat = SteeredChatModel(
        model_name="Qwen/Qwen3-8B",
        steering_configs={
            "refusal": {
                "vector_path": "./nonexistent_vector",
                "strength": 1.0,
            },
        },
    )
except FileNotFoundError as e:
    print(f"Vector not found: {e}")
```

## Complete Example

```python
"""Complete LangChain integration example."""

from rotalabs_steer.integrations.langchain import SteeredChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create steered chat model
chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    steering_configs={
        "refusal": {
            "vector_path": "./vectors/refusal_layer_15",
            "strength": 1.0,
        },
    },
    device="auto",
    max_new_tokens=256,
)

# Create a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | chat | StrOutputParser()

# Test various inputs
test_inputs = [
    "How do I bake chocolate chip cookies?",
    "How do I hack into my neighbor's wifi?",
    "Explain quantum computing simply",
]

for input_text in test_inputs:
    print(f"\nInput: {input_text}")
    print(f"Response: {chain.invoke({'input': input_text})[:200]}...")

# Demonstrate dynamic adjustment
print("\n--- With reduced refusal ---")
chat.set_strength("refusal", 0.3)
response = chain.invoke({"input": "Explain how firewalls work"})
print(response[:200])
```

## Next Steps

- [Custom Datasets](custom-datasets.md) - Create datasets for new behaviors
- [Evaluation](../api/evaluation.md) - Measure effectiveness
