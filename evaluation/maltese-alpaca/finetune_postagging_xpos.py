import os

import fire
import torch
import transformers
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv()

from peft import (
    prepare_model_for_int8_training,
    PeftModel
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from huggingface_hub import login
login(os.getenv('HF_READ'))

import warnings
warnings.filterwarnings("ignore")

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-hf",  # the only required argument
    data_path: str = "evaluation/data/pos",
    output_dir: str = "./evaluation/maltese-alpaca/maltese-alpaca-pos-xpos",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 5e-4,
    cutoff_len: int = 256,
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "maltese-alpaca",
    wandb_run_name: str = "xpos-pos-finetune",
    wandb_watch: str = "all",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            # truncation=True,
            # max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):

        instruction = 'Agħmel part-of-speech tagging ta\' din is-sentenza.'

        full_prompt = """Hawn taħt hawn struzzjoni li tiddeskrivi kompitu, flimkien ma' input li jipprovdi aktar kuntest. Ikteb tweġiba li timla t-talba kif xieraq.\n\n### Istruzzjoni: \n{instruction}\n\n### Input:\n{input}\n\n### Risposta:\n{output}""".format(instruction=instruction, input=data_point['tokens'], output=data_point['xpos'])

        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    LORA_WEIGHTS = 'kelseybonnici/maltese-alpaca'

    model = PeftModel.from_pretrained(
        model = model,
        model_id = LORA_WEIGHTS,
        is_trainable = True,
        torch_dtype=torch.float16,
    )

    data = load_dataset(data_path)
    
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = (
        data["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        data["validation"].shuffle().map(generate_and_tokenize_prompt)
    )
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_safetensors = False,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train()

    model.save_pretrained(output_dir, safe_serialization = False)
    model.push_to_hub('kelseybonnici/maltese-alpaca-pos-xpos', token = os.getenv('HF_WRITE'), safe_serialization = False)


if __name__ == "__main__":
    fire.Fire(train)