"""Training Framework Documentation

# Phase 3: Training Framework

## Overview

The Training Framework handles model optimization through backpropagation.

## Components

### 1. Optimizer: AdamW
- Adaptive learning rates per parameter
- Weight decay decoupled (not L2 regularization)
- Better than Adam for large models

```
AdamW update:
  m_t = β₁ m_{t-1} + (1 - β₁) ∇L
  v_t = β₂ v_{t-1} + (1 - β₂) (∇L)²
  θ_t = θ_{t-1} - α (m_t / (√v_t + ε) + λ θ_{t-1})
```

### 2. Loss Function: Cross-Entropy
```python
Loss = -sum(target_one_hot * log(softmax(logits))) / batch_size
```

### 3. Learning Rate Schedule
- Warmup: Linear increase from 0 to max_lr
- Decay: Cosine annealing to min_lr

### 4. Gradient Clipping
- Prevent exploding gradients
- Clip by norm

### 5. Checkpointing
- Save best model during training
- Resume from checkpoint
- Early stopping

## Training Loop

```python
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimization step
        optimizer.step()
        scheduler.step()
        
        # Logging
        if batch_idx % log_interval == 0:
            print(f"Loss: {loss.item():.4f}")
    
    # Validation
    val_loss = evaluate(model, val_loader)
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint(model, epoch)
```

## Data Processing

### Tokenization
- Convert raw text to token IDs using tokenizer
- Padding with <PAD> tokens
- Attention masks for padding

### Batching
```python
batch = {
    'input_ids': [batch_size, seq_len],
    'attention_mask': [batch_size, seq_len],
    'labels': [batch_size, seq_len]
}
```

### Sequence Packing
- Concatenate short sequences
- Use document boundaries
- Maximize GPU utilization

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 32-128 | Larger = faster but more memory |
| learning_rate | 1e-4 to 5e-4 | Depends on model size |
| weight_decay | 0.01 | L2 regularization strength |
| warmup_steps | 10K | Percentage of total steps |
| gradient_clip | 1.0 | Max gradient norm |
| epochs | 3-10 | Usually 2-4 passes sufficient |

## Validation & Evaluation

### Perplexity
```
Perplexity = exp(mean(loss))
- Measures model's uncertainty
- Lower is better
- ~10-50 on good models
```

### Generation Quality
- Human evaluation
- Automatic metrics (BLEU, ROUGE)
- Likelihood on held-out test set

## Distributed Training

### Data Parallelism
- Copy model to each GPU
- Split batch across GPUs
- Synchronize gradients
- Scales nearly linearly

### Model Parallelism
- Split model layers across GPUs
- For very large models
- Pipeline parallelism

## Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler()

with torch.autocast(device_type='cuda', dtype=torch.float16):
    logits = model(inputs)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

## Troubleshooting

### Loss not decreasing
- Learning rate too low/high
- Gradient clipping too aggressive
- Bad initialization

### NaN/Inf loss
- Learning rate too high
- Numerical instability
- Use mixed precision

### Out of memory
- Reduce batch size
- Use gradient checkpointing
- Use mixed precision
- Enable activation recomputation

---

**Status**: Design Complete - Ready for Implementation
"""
