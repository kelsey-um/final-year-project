import torch
from peft import PeftModel
import transformers
import gradio as gr
import json
import pandas as pd

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

BASE_MODEL = "huggyllama/llama-7b"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Hawn taħt hawn struzzjoni li tiddeskrivi kompitu, flimkien ma' input li jipprovdi aktar kuntest. Ikteb tweġiba li timla t-talba kif xieraq.\n\n### Istruzzjoni: \n{instruction}\n\n### Input:\n{input}\n\n### Risposta:\n"""
    else:
        return f"""Hawn taħt hawn struzzjoni li tiddeskrivi kompitu. Ikteb tweġiba li tikkompleta t-talba kif xieraq.\n\n### Istruzzjoni:\n{instruction}\n\n### Risposta:\n"""


model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    with open("data/", "rb") as json_file: #CHANGE FILE NAME
        json_data = json.loads(json_file.read())
    df = pd.DataFrame(json_data)
    
    output_list = []
    for index, row in df.iterrows():
        output_list.append(evaluate(row['instruction'], row['input']))

    df["output"] = pd.Series(output_list)
    
    translated_dict = df.to_dict('records')

    with open(f"data/results.json", 'w') as file:
        json.dump(translated_dict, file)