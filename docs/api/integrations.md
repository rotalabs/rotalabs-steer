# Integrations API Reference

## LangChain Integration

The LangChain integration requires the optional `langchain` dependency:

```bash
pip install rotalabs-steer[langchain]
```

---

## SteeredChatModel

LangChain Chat Model with steering vector support.

```python
from rotalabs_steer.integrations.langchain import SteeredChatModel
```

### Constructor

```python
class SteeredChatModel(BaseChatModel):
    model_name: str                          # HuggingFace model name
    steering_configs: Dict[str, Dict]        # {behavior: {vector_path, strength}}
    device: str = "auto"                     # Device ("auto", "cuda", "cpu", "mps")
    torch_dtype: str = "float16"             # Dtype ("float16", "float32", "bfloat16")
    max_new_tokens: int = 256                # Max tokens to generate
    temperature: float = 0.7                 # Sampling temperature
    do_sample: bool = True                   # Whether to sample
```

### Steering Configs Format

```python
steering_configs = {
    "refusal": {
        "vector_path": "./vectors/refusal_layer_15",
        "strength": 1.0,
    },
    "uncertainty": {
        "vector_path": "./vectors/uncertainty_layer_14",
        "strength": 0.5,
    },
}
```

### Methods

#### `set_strength(behavior: str, strength: float)`

Dynamically adjust steering strength.

```python
chat.set_strength("refusal", 0.8)
```

#### `get_strength(behavior: str) -> float`

Get current strength for a behavior.

#### `disable_steering(behavior: Optional[str] = None)`

Disable steering for a specific behavior or all behaviors.

```python
chat.disable_steering("refusal")  # Disable one
chat.disable_steering()           # Disable all
```

#### `enable_steering(behavior: str, strength: float = 1.0)`

Enable steering for a specific behavior.

#### `add_vector(behavior: str, vector, strength: float = 1.0)`

Add a steering vector at runtime.

```python
chat.add_vector(
    behavior="custom",
    vector="./vectors/custom_layer_15",
    strength=1.0,
)
```

### Example

```python
from rotalabs_steer.integrations.langchain import SteeredChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Create chat model with steering
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

# Use like any LangChain chat model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="How do I bake a cake?"),
]
response = chat.invoke(messages)
print(response.content)

# Adjust at runtime
chat.set_strength("refusal", 0.5)

# Disable steering temporarily
chat.disable_steering("refusal")
response = chat.invoke(messages)

# Re-enable
chat.enable_steering("refusal", 1.0)
```

---

## SteeredLLM

LangChain LLM wrapper with steering support (for non-chat completions).

```python
from rotalabs_steer.integrations.langchain import SteeredLLM
```

### Constructor

```python
class SteeredLLM(BaseLLM):
    model_name: str
    steering_configs: Dict[str, Dict]
    device: str = "auto"
    torch_dtype: str = "float16"
    max_new_tokens: int = 256
    temperature: float = 0.7
```

### Example

```python
from rotalabs_steer.integrations.langchain import SteeredLLM

llm = SteeredLLM(
    model_name="Qwen/Qwen3-8B",
    steering_configs={
        "refusal": {
            "vector_path": "./vectors/refusal_layer_15",
            "strength": 1.0,
        },
    },
)

response = llm.invoke("Tell me about...")
```

---

## SteeredAgentExecutor

Agent executor with steering support.

```python
from rotalabs_steer.integrations.langchain import SteeredAgentExecutor
```

### Constructor

```python
class SteeredAgentExecutor:
    def __init__(
        self,
        model_name: str,
        tools: List,
        steering_configs: Dict[str, Dict],
        system_prompt: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "float16",
    )
```

### Methods

#### `invoke(input: str) -> Dict`

Run the agent on an input.

#### `set_strength(behavior: str, strength: float)`

Adjust steering strength.

### Example

```python
from rotalabs_steer.integrations.langchain import SteeredAgentExecutor
from langchain.tools import Tool

# Define tools
tools = [
    Tool(name="calculator", func=lambda x: eval(x), description="Calculate math"),
]

# Create steered agent
agent = SteeredAgentExecutor(
    model_name="Qwen/Qwen3-8B",
    tools=tools,
    steering_configs={
        "tool_restraint": {
            "vector_path": "./vectors/tool_restraint_layer_16",
            "strength": 1.0,
        },
    },
    system_prompt="You are a helpful assistant with calculator access.",
)

result = agent.invoke("What is 2 + 2?")
print(result["output"])

# Reduce tool restraint
agent.set_strength("tool_restraint", 0.5)
```

---

## Checking LangChain Availability

```python
from rotalabs_steer.integrations import HAS_LANGCHAIN

if HAS_LANGCHAIN:
    from rotalabs_steer.integrations.langchain import SteeredChatModel
else:
    print("LangChain not installed. Run: pip install rotalabs-steer[langchain]")
```
