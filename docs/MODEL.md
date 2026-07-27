"""Foundation Model Architecture Documentation

# Phase 2: Foundation Model (Transformer Decoder)

## Overview

The Foundation Model is the **core neural engine** of Project Aura. It uses a Transformer decoder architecture for autoregressive text generation.

**Architecture Type**: Transformer Decoder (like GPT)

## Components

### 1. Token Embeddings
- Maps token IDs → dense vectors
- Size: vocab_size × embedding_dim
- Learned during training

### 2. Positional Embeddings
- Encodes position information in sequences
- Absolute positional encoding (learnable)
- Alternative: Rotary embeddings for better extrapolation

### 3. Self-Attention Layers
- Multi-head attention mechanism
- Allows tokens to attend to previous tokens only (causal masking)
- Learn relationships between tokens

### 4. Feed-Forward Networks
- Dense-ReLU-Dense layers
- Applied to each position independently
- Provides non-linearity and capacity

### 5. Layer Normalization
- Stabilizes training
- Pre-norm or post-norm architecture

### 6. Output Projection
- Projects final hidden states to vocab logits
- Shared with embedding matrix (weight tying)

## Model Configuration

Typical small model:
```python
config = {
    'vocab_size': 10000,
    'embedding_dim': 768,      # Hidden dimension
    'num_layers': 12,          # Number of transformer blocks
    'num_heads': 12,           # Attention heads
    'ffn_dim': 3072,          # Feed-forward hidden dim (usually 4x embedding_dim)
    'max_seq_len': 2048,      # Maximum sequence length
    'dropout': 0.1,            # Dropout rate
}
```

## Forward Pass

```
1. Input: token_ids [batch_size, seq_len]

2. Embedding: [batch_size, seq_len, embedding_dim]
   embedded = embedding(token_ids) + positional_embedding(positions)

3. For each of N transformer blocks:
   a) Multi-head self-attention
      attn_out = attention(embedded, embedded, embedded, mask=causal)
      embedded = layer_norm(embedded + attn_out)  # Residual
   
   b) Feed-forward
      ffn_out = linear(relu(linear(embedded)))
      embedded = layer_norm(embedded + ffn_out)    # Residual

4. Output projection: [batch_size, seq_len, vocab_size]
   logits = output_projection(embedded)

5. Loss: cross_entropy(logits, targets)
```

## Inference (Generation)

```python
model.eval()

# Start with BOS token
token_ids = [BOS_TOKEN_ID]
context_len = 1

while len(token_ids) < max_length:
    # Get logits for next token
    logits = model(token_ids[-context_len:])
    next_logits = logits[-1, :]  # Last position
    
    # Sample next token (with temperature)
    next_token = sample(next_logits, temperature=0.7)
    
    if next_token == EOS_TOKEN_ID:
        break
    
    token_ids.append(next_token)
    context_len = min(context_len + 1, max_seq_len)

text = tokenizer.decode(token_ids)
```

## Key Implementation Details

### Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-head:
  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
  MultiHead = Concat(head_1, ..., head_h) W^O
```

### Causal Masking
- Prevents attention to future tokens
- Essential for autoregressive generation
- Implemented as: logits[i, j] = -inf if j > i

### Residual Connections
- x' = x + SubLayer(LayerNorm(x))
- Helps gradient flow in deep networks

### Layer Normalization
- Normalizes activations: (x - mean) / sqrt(var + eps)
- Stabilizes training
- Reduces internal covariate shift

## Parameter Count

For configuration above:
```
Total params ≈ (vocab_size × embedding_dim) +          # Embedding
                (embedding_dim × embedding_dim × 4) +    # Attention proj
                (embedding_dim × ffn_dim × 2) +           # FFN
                (embedding_dim) × num_layers × 2          # Layer norms
              × num_layers

            ≈ 10K × 768 + (4 × 768² + 2 × 768 × 3072 + 2 × 768) × 12
            ≈ 7.68M + 28M × 12
            ≈ 336M parameters
```

## Training Strategy

### Next Token Prediction
- **Objective**: Predict next token given previous tokens
- **Loss**: Cross-entropy between predicted logits and target tokens
- **Data**: Autoregressive: position i predicts position i+1

### Sequence Processing
```
Input:  [<BOS>, "The", "cat", "sat"]
Target: ["The", "cat", "sat", "<EOS>"]
Loss:   computed only on target positions
```

### Batching
- Pack multiple sequences in batch
- Pad shorter sequences
- Attention masks account for padding

## Optimization Tips

1. **Gradient Accumulation**: Update only every N batches to simulate larger batch
2. **Mixed Precision**: Use float16 for speed, keep float32 for stability
3. **Checkpointing**: Recompute activations during backprop to save memory
4. **Distributed Training**: Multi-GPU data parallelism or model parallelism
5. **Learning Rate Scheduling**: Warmup then cosine decay

## Integration Points

- **Phase 1 (Tokenizer)**: Input tokenization, output decoding
- **Phase 3 (Training)**: Model weights, optimizer
- **Phase 4 (Runtime)**: Inference engine calls model.forward()
- **Phase 5 (Memory)**: Stores embeddings for context
- **Phase 10 (Vision)**: Multimodal: image tokens + text tokens

---

**Status**: Design Complete - Ready for Implementation
"""
