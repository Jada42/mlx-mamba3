import mlx.core as mx
from .cache import MambaCache

def sample(logits, temp=0.0):
    """Sample from the logits given a temperature."""
    if temp == 0.0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits / temp)


def generate_step(model, prompt_tokens, temp=0.0, max_tokens=100):
    """
    Generator that yields tokens one by one.
    Runs prefill on prompt_tokens and then runs step recurrence autoregressively.
    """
    if isinstance(prompt_tokens, list):
        prompt_tokens = mx.array([prompt_tokens])
    elif len(prompt_tokens.shape) == 1:
        prompt_tokens = mx.expand_dims(prompt_tokens, 0)

    B, L = prompt_tokens.shape

    # Allocate inference cache
    cache = MambaCache.from_model(model, batch_size=B)

    # 1. Prefill pass: run the model over the full prompt to populate the cache
    logits = model(prompt_tokens, cache=cache)

    # Get the logits for the last token in the prompt
    last_logits = logits[:, -1, :]  # (B, vocab_size)

    next_token = sample(last_logits, temp)
    yield next_token

    # 2. Decoding loop
    curr_token = next_token
    for _ in range(max_tokens - 1):
        logits = model.step(curr_token, cache=cache)
        next_token = sample(logits, temp)
        yield next_token
        curr_token = next_token


def generate(model, prompt_tokens, temp=0.0, max_tokens=100):
    """Generate a sequence of tokens from a prompt."""
    tokens = []
    for t in generate_step(model, prompt_tokens, temp=temp, max_tokens=max_tokens):
        tokens.append(t)
    return mx.stack(tokens, axis=1)
