import os
import gc
import pandas as pd
import torch
import huggingface_hub
from datasets import Dataset

from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


class CharacterChatBot:
    """
    CharacterChatBot for Walter White using:
    - TRL 0.22.0.dev0 SFTTrainer API
    - LoRA on Llama-3-8B
    - Hugging Face pipeline for inference
    """

    def __init__(self, model_path: str, data_path: str = "/content/data/BB_data.csv",
                 huggingface_token: str | None = None):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)

        # If an adapter/model with this repo id already exists, load it; else train and push
        if huggingface_hub.repo_exists(self.model_path):
            self.pipeline = self.load_model(self.model_path)
        else:
            print("Model not found in Hugging Face Hub. Training a new model...")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.pipeline = self.load_model(self.model_path)

    # ---------------------------
    # Chat interface
    # ---------------------------
    def chat(self, message: str, history: list[list[str]]):
        tokenizer = self.pipeline.tokenizer
        messages = [{"role": "system",
                     "content": "You are Walter White from Breaking Bad. Respond in character.\n"}]
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False
        )
        return outputs[0]["generated_text"].strip()

    # ---------------------------
    # Load model + adapter
    # ---------------------------
    def load_model(self, model_or_adapter_repo: str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Try to load the base model first (QLoRA quantized)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # If a LoRA adapter repo is provided (and not just the base), load it on top
        if model_or_adapter_repo != self.base_model_path:
            try:
                model = PeftModel.from_pretrained(model, model_or_adapter_repo)
            except Exception:
                # If they pushed a full merged model instead of an adapter, fall back to loading it directly
                model = AutoModelForCausalLM.from_pretrained(
                    model_or_adapter_repo,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_repo, use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.float16}
        )

    # ---------------------------
    # Training function
    # ---------------------------
    def train(
        self,
        base_model_name_or_path: str,
        dataset: Dataset,
        output_dir: str = "./results",
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        optim: str = "paged_adamw_32bit",
        save_steps: int = 200,
        logging_steps: int = 10,
        learning_rate: float = 2e-4,
        max_grad_norm: float = 0.3,
        max_steps: int = 300,
        warmup_ratio: float = 0.3,
        lr_scheduler_type: str = "constant",
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # Needed for gradient checkpointing + QLoRA training
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            report_to="none",
            gradient_checkpointing=True,
            packing=False,
            completion_only_loss=False,  # CRITICAL: Disable to use formatting_func
        )

        # TRL 0.22 API - tokenizer is passed via the training_args, not directly
        def formatting_func(examples):
            texts = []
            for p, c in zip(examples["prompt"], examples["completion"]):
                texts.append(f"{p}{c}")
            return texts

        # FIXED: Remove tokenizer from SFTTrainer initialization
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            formatting_func=formatting_func,
            # tokenizer parameter removed - it's now handled internally
        )

        # Set tokenizer manually after initialization (if needed)
        trainer.tokenizer = tokenizer

        trainer.train()

        adapter_dir = "final_ckpt"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Reload clean base and push adapter so the hub repo is a LoRA adapter
        base_model_push = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        peft_model = PeftModel.from_pretrained(base_model_push, adapter_dir)
        peft_model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del trainer, model, base_model_push, peft_model
        gc.collect()

    # ---------------------------
    # Data loading
    # ---------------------------
    def load_data(self) -> Dataset:
        df = pd.read_csv(self.data_path).dropna()
        df["text"] = df["text"].astype(str)
        df["number_of_words"] = df["text"].str.strip().str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        df["walter_response_flag"] = ((df["actor"] == "Walter") & (df["number_of_words"] > 5)).astype(int)

        idxs = df.index[df["walter_response_flag"] == 1].tolist()
        system_prompt = "You are Walter White from Breaking Bad. Respond in his style.\n"

        prompts, completions = [], []
        for i in idxs:
            user_line = str(df.iloc[i - 1]["text"]).strip() if i - 1 in df.index else ""
            walter_line = str(df.iloc[i]["text"]).strip()
            prompt_text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_line}\n<|assistant|>\n"
            prompts.append(prompt_text)
            completions.append(walter_line)

        dataset = Dataset.from_pandas(pd.DataFrame({"prompt": prompts, "completion": completions}))
        return dataset


