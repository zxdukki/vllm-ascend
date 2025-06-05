import os
import time

from vllm import LLM, SamplingParams

# enable dual-batch overlap for vllm ascend
os.environ["VLLM_ASCEND_ENABLE_DBO"] = "1"
os.environ["VLLM_USE_V1"] = "1"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 10
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        hf_overrides={
            "architectures": ["DeepseekDBOForCausalLM"],
        },  # override the model arch to run the dbo model
        enforce_eager=True,
        tensor_parallel_size=8,
        max_num_seqs=16,
        max_model_len=8192,
        max_num_batched_tokens=32768,
        block_size=128,
        compilation_config=1,
        gpu_memory_utilization=0.96)

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
