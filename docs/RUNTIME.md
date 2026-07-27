"""Runtime Engine Documentation

# Phase 4: Runtime Engine

## Overview

The Runtime Engine executes inference: taking input text and generating output text token-by-token.

## Components

### 1. Inference Engine
- Manages generation state
- Caches key-value pairs for efficiency
- Handles context management

### 2. Sampler
- Temperature scaling
- Top-K filtering
- Top-P (nucleus) sampling
- Greedy decoding

### 3. Context Manager
- Maintains KV cache
- Manages sequence position
- Handles padding

## Generation Methods

### Greedy Decoding
```python
for t in range(max_length):
    logits = model(input_ids)
    next_token = argmax(logits[-1, :])
    input_ids.append(next_token)
```
Fast but can get stuck in loops.

### Temperature Sampling
```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```
temperature < 1.0: More deterministic (sharper dist)
temperature > 1.0: More random (softer dist)

### Top-K Sampling
```python
keep_top_k = topk(logits, k)
logits[~keep_top_k] = -inf
probs = softmax(logits)
next_token = sample(probs)
```
Only considers K most likely tokens.

### Top-P (Nucleus) Sampling
```python
sorted_logits = sort(logits, descending=True)
cumsum_probs = cumsum(softmax(sorted_logits))
cutoff_idx = where(cumsum_probs > p)[0][0]
logits[logits < sorted_logits[cutoff_idx]] = -inf
probs = softmax(logits)
next_token = sample(probs)
```
Keeps tokens until cumulative probability > p.

## KV-Cache

Optimization to avoid recomputing attention for all previous tokens:

```python
# Without cache (slow):
for i in range(n):
    logits = model(input_ids[:i+1])  # Recomputes all i+1 positions

# With cache (fast):
kv_cache = None
for i in range(n):
    logits, kv_cache = model(input_ids[i:i+1], kv_cache=kv_cache)
```

Speedup: ~100x for long sequences

## Inference Loop

```python
def generate(prompt, max_length=100, temperature=1.0, top_p=0.9):
    # Tokenize input
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    generated = input_ids.copy()
    
    kv_cache = None
    
    for _ in range(max_length):
        # Model forward pass
        logits, kv_cache = model(
            input_ids=torch.tensor([generated[-1:]]),
            kv_cache=kv_cache
        )
        
        # Apply temperature
        logits = logits[-1, :] / temperature
        
        # Apply top-p
        logits = apply_top_p(logits, top_p)
        
        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        if next_token == EOS_TOKEN:
            break
        
        generated.append(next_token.item())
    
    # Decode output
    return tokenizer.decode(generated)
```

## Performance Optimization

### Batching
- Generate multiple sequences in parallel
- Pad to same length
- 10x faster than sequential

### Model Compilation
- JIT compile model
- Fuse operations
- 1.5-2x speedup

### Quantization
- INT8 or INT4 inference
- 4x speedup, slight quality loss

### Distillation
- Smaller student model
- Knowledge transfer from teacher

## Stopping Criteria

```python
stop_conditions = [
    reached_max_length,
    generated_eos_token,
    generated_stop_sequence,
    timeout_exceeded,
]
```

## Integration

- **Tokenizer**: Input text → tokens, tokens → output text
- **Model**: Executes forward pass
- **Memory**: Stores conversation history
- **Reasoning**: Chains generation calls

---

**Status**: Design Complete - Ready for Implementation
"""
