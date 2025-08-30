from vllm import LLM, SamplingParams

def main() -> None:
    # Initialize the LLM with the desired model
    # example: https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
    llm = LLM(model="Qwen/Qwen3-32B", tensor_parallel_size=1, trust_remote_code=True)

    # Run inference with a sample prompt
    # MATH-500 dataset example
    # https://huggingface.co/datasets/HuggingFaceH4/MATH-500/viewer/default/test?views%5B%5D=test
    # unique_id: test/intermediate_algebra/1994.json
    prompt = r"""Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]in terms of $p$ and $q.$"""

    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=0.6,
    )

    response = llm.generate(prompt, sampling_params=sampling_params)
    
    # Print the response
    # Answer should be: p - q
    print("==" * 20)
    print(response[0].outputs[0].text)



if __name__ == "__main__":
    main()
