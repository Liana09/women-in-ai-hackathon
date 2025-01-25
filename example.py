from vllm import LLM
from vllm.sampling_params import SamplingParams
from dotenv import load_dotenv
import torch

# Debugging MPS support
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Model configuration
model_name = "mistralai/Pixtral-12B-2409"
load_dotenv()
sampling_params = SamplingParams(max_tokens=8192)

# Initialize the LLM with enforce_eager=True for synchronous processing
from vllm import LLM

llm = LLM(
    model=model_name,
    device="cpu",  # Use CPU instead of MPS
)


# Prepare prompt and messages
print("There")
prompt = "Describe this image in one sentence."
image_url = "https://picsum.photos/id/237/200/300"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}, 
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    },
]

# Generate outputs
outputs = llm.chat(messages, sampling_params=sampling_params)

# Print output
print(outputs[0].outputs[0].text)
