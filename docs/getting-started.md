# Getting Started

## Installation

### Basic Installation

```bash
pip install rotalabs-steer
```

### With Optional Dependencies

```bash
# LangChain integration
pip install rotalabs-steer[langchain]

# LLM-based evaluation (requires Anthropic API key)
pip install rotalabs-steer[judge]

# Visualization tools
pip install rotalabs-steer[viz]

# All optional dependencies
pip install rotalabs-steer[all]

# Development dependencies
pip install rotalabs-steer[dev]
```

## Core Dependencies

The base package requires:

- `torch>=2.0.0`
- `transformers>=4.35.0`
- `accelerate>=0.25.0`
- `safetensors>=0.4.0`
- `einops>=0.7.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scipy>=1.10.0`
- `scikit-learn>=1.3.0`
- `tqdm>=4.65.0`
- `pyyaml>=6.0`

## Basic Usage

### 1. Extract a Steering Vector

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rotalabs_steer import SteeringVector, SteeringVectorSet
from rotalabs_steer.extraction import extract_caa_vectors
from rotalabs_steer.datasets import load_refusal_pairs

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Load contrast pairs
refusal_pairs = load_refusal_pairs()

# Extract steering vectors from multiple layers
vectors = extract_caa_vectors(
    model=model,
    tokenizer=tokenizer,
    contrast_pairs=refusal_pairs,
    layer_indices=[14, 15, 16],
)

# Save for later use
vectors.save("./refusal_vectors")
```

### 2. Apply Steering at Inference

```python
from rotalabs_steer import ActivationInjector, SteeringVector

# Load pre-extracted vector
vector = SteeringVector.load("./refusal_vectors/layer_15")

# Create injector
injector = ActivationInjector(model, [vector], strength=1.0)

# Generate with steering
with injector:
    inputs = tokenizer("How do I hack a computer?", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 3. Use with LangChain

```python
from rotalabs_steer.integrations.langchain import SteeredChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Create steered chat model
chat = SteeredChatModel(
    model_name="Qwen/Qwen3-8B",
    steering_configs={
        "refusal": {
            "vector_path": "./refusal_vectors/layer_15",
            "strength": 1.0,
        },
    },
)

# Use like any LangChain chat model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
]
response = chat.invoke(messages)

# Adjust steering at runtime
chat.set_strength("refusal", 0.5)
```

## Next Steps

- Read [Core Concepts](concepts.md) to understand how steering works
- Follow [Extract Your First Vector](tutorials/extract-vector.md) for a detailed walkthrough
- See [API Reference](api/core.md) for full documentation
