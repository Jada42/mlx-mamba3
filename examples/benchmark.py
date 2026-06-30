import time
import mlx.core as mx
from mlx_mamba_native.model import MambaLMHeadModel, MambaConfig
from mlx_mamba_native.generate import generate

def run_benchmark():
    # 1. Setup a standard micro configuration
    config = MambaConfig(
        d_model=256,
        n_layer=4,
        vocab_size=2000,
        ssm_cfg={"is_mimo": True, "mimo_rank": 4}
    )
    print("Initializing Mamba-3 model for benchmark...")
    model = MambaLMHeadModel(config)

    # Warm-up (triggers JIT compilation)
    print("Warming up JIT compiler...")
    prompt = [1, 2, 3, 4]
    _ = generate(model, prompt, temp=0.0, max_tokens=10)
    mx.eval(model.parameters())

    # 2. Benchmark Prefill
    prompt_len = 128
    prompt_large = list(range(prompt_len))
    
    print(f"Benchmarking prefill on sequence of length {prompt_len}...")
    # First run (compiles the sequence forward graph)
    t_start = time.perf_counter()
    logits = model(mx.array([prompt_large]))
    mx.eval(logits)
    compile_and_prefill = (time.perf_counter() - t_start) * 1000
    
    # Second run (fully compiled execution)
    t_start = time.perf_counter()
    logits = model(mx.array([prompt_large]))
    mx.eval(logits)
    prefill_compiled = (time.perf_counter() - t_start) * 1000
    
    print(f"  First Run (JIT Compile + Prefill): {compile_and_prefill:.2f} ms")
    print(f"  Subsequent Runs (Compiled Prefill): {prefill_compiled:.2f} ms")

    # 3. Benchmark Decode (tokens/sec)
    num_tokens = 100
    print(f"Benchmarking decode of {num_tokens} tokens...")
    t0 = time.perf_counter()
    output = generate(model, prompt_large, temp=0.0, max_tokens=num_tokens)
    mx.eval(output)
    total_time = time.perf_counter() - t0
    
    # We subtract prefill time to get pure decode throughput
    decode_time = total_time - (prefill_compiled / 1000.0)
    tok_per_sec = num_tokens / decode_time
    print(f"  Decode Throughput: {tok_per_sec:.2f} tok/s")
    print(f"  Total generation time: {total_time:.2f} s")

if __name__ == "__main__":
    run_benchmark()
