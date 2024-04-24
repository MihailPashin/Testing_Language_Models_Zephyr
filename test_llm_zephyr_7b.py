import os
import urllib.request
from llama_cpp import Llama


def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")


# Dowloading GGML model from HuggingFace
ggml_model_path = "https://huggingface.co/TheBloke/OpenOrca-Zephyr-7B-GGUF/resolve/main/openorca-zephyr-7b.Q6_K.gguf?download=true"
filename = "openorca-zephyr-7b.Q6_K.gguf"

download_file(ggml_model_path, filename)


llm = Llama(model_path="openorca-zephyr-7b.Q6_K.gguf", n_ctx=4096, n_batch=156)


def generate_text(
    prompt="Hello! How are you today?",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text


def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|system|>
You're a psychologist. You're asking me questions: What and Who is important to user? What are troubles inside in user? What do user do for reliefing the pain? What do user do for moving towards user's goals and values?
</s>
<|user|>
{input}</s>
<|assistant|>
"""
    return chat_prompt_template


prompt = generate_prompt_from_template(
    "Hello! I am on your session."
)

s= generate_text(
    prompt,
    max_tokens=256,
)
print(s)