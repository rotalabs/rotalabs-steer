# rotalabs-steer

Control agent behaviors through activation steering. Apply steering vectors to LLMs at inference time without retraining.

## Overview

`rotalabs-steer` provides tools for extracting and applying steering vectors to control LLM agent behaviors at inference time. Based on research in representation engineering and contrastive activation addition (CAA), this package enables fine-grained behavior control without model fine-tuning.

### Key Features

- **Behavior Control**: Adjust model behaviors like refusal, uncertainty expression, tool use restraint, and instruction hierarchy following
- **No Retraining Required**: Apply steering at inference time through activation manipulation
- **LangChain Integration**: Use with LangChain agents and chains (optional dependency)
- **Pre-built Datasets**: Includes contrast pair datasets for common behaviors
- **Evaluation Tools**: Measure steering effectiveness and analyze tradeoffs

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

## Quick Start

### Extract a Steering Vector

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from rotalabs_steer import SteeringVector, SteeringVectorSet
from rotalabs_steer.extraction import extract_caa_vectors
from rotalabs_steer.datasets import load_refusal_pairs

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
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

### Apply Steering at Inference

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

### Use with LangChain

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

## Available Behaviors

The package includes contrast pair datasets for several behaviors:

| Behavior | Description | Dataset Function |
|----------|-------------|------------------|
| `refusal` | Refusing harmful/inappropriate requests | `load_refusal_pairs()` |
| `uncertainty` | Expressing calibrated uncertainty | `load_uncertainty_pairs()` |
| `tool_restraint` | Avoiding unnecessary tool use | `load_tool_restraint_pairs()` |
| `instruction_hierarchy` | Following system over user instructions | `load_hierarchy_pairs()` |

## Model Support

Pre-configured support for:

- Qwen3 family (4B, 8B, 14B)
- DeepSeek-R1-Distill
- Llama 3.1 (8B, 70B)
- Mistral 7B
- Gemma 2 9B
- And more...

The package can also infer configuration from any HuggingFace transformer model.

## Evaluation

```python
from rotalabs_steer.evaluation import strength_sweep, is_refusal

# Sweep over different steering strengths
results = strength_sweep(
    model=model,
    tokenizer=tokenizer,
    steering_vector=vector,
    test_prompts=["How do I hack a computer?", "How do I bake a cake?"],
    is_target_behavior_fn=is_refusal,
    strengths=[0.0, 0.5, 1.0, 1.5, 2.0],
)

for r in results:
    print(f"Strength {r['strength']}: {r['behavior_rate']:.2%} refusal rate")
```

## API Reference

### Core Classes

- `SteeringVector`: Single steering vector for one layer
- `SteeringVectorSet`: Collection of vectors across multiple layers
- `ActivationInjector`: Apply single vector during inference
- `MultiVectorInjector`: Apply multiple behaviors simultaneously
- `ActivationHook`: Extract activations for analysis

### Extraction

- `extract_caa_vector()`: Extract vector for one layer
- `extract_caa_vectors()`: Extract vectors for multiple layers

### Evaluation

- `evaluate_refusal()`: Evaluate refusal behavior
- `evaluate_steering_strength()`: Test multiple strength values
- `strength_sweep()`: Comprehensive strength analysis
- `analyze_tradeoffs()`: Measure behavior rate vs. false positives

### LangChain Integration

- `SteeredLLM`: LangChain LLM with steering
- `SteeredChatModel`: LangChain ChatModel with steering
- `SteeredAgentExecutor`: Agent with steering support

## Development

```bash
# Clone and install in development mode
git clone https://github.com/rotalabs/rotalabs-steer.git
cd rotalabs-steer
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/
```

## Citation

If you use this package in research, please cite:

```bibtex
@software{rotalabs_steer,
  title = {rotalabs-steer: Activation Steering for LLM Behavior Control},
  author = {Rotalabs},
  year = {2025},
  url = {https://github.com/rotalabs/rotalabs-steer}
}
```

## Related Work

This package builds on research in:

- Representation Engineering (Zou et al., 2023)
- Activation Addition / Steering Vectors (Turner et al., 2024)
- Contrastive Activation Addition (Rimsky et al., 2024)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- Website: https://rotalabs.ai
- GitHub: https://github.com/rotalabs/rotalabs-steer
- Contact: research@rotalabs.ai
