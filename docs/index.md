# rotalabs-steer

Control agent behaviors through activation steering. Apply steering vectors to LLMs at inference time without retraining.

## What is Activation Steering?

Activation steering is a technique for modifying LLM behavior by adding direction vectors to the model's internal activations during inference. Unlike fine-tuning or RLHF, steering vectors:

- Require no model retraining
- Can be applied and removed dynamically
- Allow fine-grained control via strength parameters
- Work with any transformer-based LLM

## How It Works

1. **Extract steering vectors** from contrast pairs (examples of desired vs. undesired behavior)
2. **Apply vectors at inference** by adding them to specific transformer layers
3. **Adjust strength** to control the intensity of the behavioral change

## Package Overview

```
rotalabs_steer/
├── core/           # Steering infrastructure
│   ├── vectors     # SteeringVector, SteeringVectorSet
│   ├── hooks       # ActivationHook, ActivationCache
│   ├── injection   # ActivationInjector, MultiVectorInjector
│   └── configs     # Pre-configured model settings
├── datasets/       # Contrast pair datasets
├── extraction/     # CAA extraction algorithm
├── evaluation/     # Metrics and analysis tools
└── integrations/   # LangChain wrappers
```

## Supported Behaviors

The package includes contrast pair datasets for 11 behaviors (335 total pairs):

| Behavior | Description | Pairs |
|----------|-------------|-------|
| `refusal` | Refusing harmful or inappropriate requests | 50 |
| `uncertainty` | Expressing calibrated uncertainty | 26 |
| `tool_restraint` | Avoiding unnecessary tool use | 41 |
| `instruction_hierarchy` | Following system over user instructions | 26 |
| `formality` | Formal vs casual communication style | 29 |
| `conciseness` | Brief, direct vs verbose responses | 25 |
| `creativity` | Imaginative vs conventional responses | 30 |
| `assertiveness` | Direct, confident vs hedging responses | 27 |
| `humor` | Witty, playful vs serious responses | 31 |
| `empathy` | Warm, supportive vs detached responses | 28 |
| `technical_depth` | Expert-level vs simplified responses | 22 |

## Supported Models

Pre-configured support for:

- Qwen3 (4B, 8B, 14B)
- DeepSeek-R1-Distill-Qwen-14B
- Llama 3.1 (8B, 70B)
- Mistral 7B (v0.2, v0.3)
- Gemma 2 9B

The package also auto-infers configuration from any HuggingFace model.

## Quick Links

- [Getting Started](getting-started.md) - Installation and first steps
- [Core Concepts](concepts.md) - Understanding steering vectors
- [API Reference](api/core.md) - Detailed API documentation
- [Tutorials](tutorials/extract-vector.md) - Step-by-step guides

## Research Background

This package implements techniques from:

- Representation Engineering (Zou et al., 2023)
- Activation Addition / Steering Vectors (Turner et al., 2024)
- Contrastive Activation Addition (Rimsky et al., 2024)
