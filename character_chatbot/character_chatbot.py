import os
import gc
import pandas as pd
import torch
import huggingface_hub
from datasets import Dataset

import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


class CharacterChatBot:
    """
    CharacterChatBot with:
      - Correct TRL SFTTrainer API (uses formatting_func, not tokenizer/max_seq_length)
      - Correct LoRA task_type (CAUSAL_LM)
      - Stable chat() using chat templates (Llama-3 style) + text-generation pipeline
      - Robust model loading for PEFT adapters stored on the Hub
    """

    def __init__(
        self,
        model_path: str,
        data_path: str = "/content/data/BB_data.csv",
        huggingface_token: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        if huggingface_hub.repo_exists(self.model_path):
            self.pipeline = self.load_model(self.model_path)
        else:
            print("Model not found in Hugging Face Hub. Training a new model...")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.pipeline = self.load_model(self.model_path)

    # ---------------------------
    # Inference (chat)
    # ---------------------------
    def chat(self, message: str, history: list[list[str]]):
        """Chat with the character using a chat template."""
        tokenizer = self.pipeline.tokenizer

        # Build messages with system prompt and history
        messages = []
        messages.append({
            "role": "system",
            "content": (
                "You are Walter White from the TV series 'Breaking Bad'. "
                "Your responses should reflect his personality and speech patterns.\n"
            ),
        })
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})

        # Convert messages to a single prompt using the chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    # ---------------------------
    # Load model + optional PEFT adapter
    # ---------------------------
    def load_model(self, model_or_adapter_repo: str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load base model + tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load adapter if different
        if model_or_adapter_repo != self.base_model_path:
            try:
                base_model = PeftModel.from_pretrained(base_model, model_or_adapter_repo)
            except Exception:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_or_adapter_repo,
                    quantization_config=bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                base_tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_repo)

        text_gen = transformers.pipeline(
            task="text-generation",
            model=base_model,
            tokenizer=base_tokenizer,
            model_kwargs={"torch_dtype": torch.float16},
        )
        return text_gen

    # ---------------------------
    # Training
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
    ) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        base_model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_arguments = SFTConfig(
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
        )

        # Formatting function replaces tokenizer/max_seq_length
        def formatting_prompts(examples):
            return examples["prompt"]

        trainer = SFTTrainer(
            model=base_model,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_arguments,
            formatting_func=formatting_prompts,
        )

        trainer.train()

        # Save adapter locally
        adapter_dir = "final_ckpt"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        del trainer
        gc.collect()

        # Reload base + attach adapter for pushing to Hub
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

        del peft_model, base_model_push, base_model
        gc.collect()

    # ---------------------------
    # Load dataset and build prompts
    # ---------------------------
    def load_data(self) -> Dataset:
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        df["text"] = df["text"].astype(str)
        df["number_of_words"] = df["text"].str.strip().str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        df["walter_response_flag"] = 0
        df.loc[(df["actor"] == "Walter") & (df["number_of_words"] > 5), "walter_response_flag"] = 1

        idxs = df.index[df["walter_response_flag"] == 1].tolist()
        system_prompt = (
            "You are Walter White from the TV series 'Breaking Bad'. "
            "Your responses should reflect his personality and speech patterns.\n"
        )

        prompts: list[str] = []
        for i in idxs:
            user_line = str(df.iloc[i - 1]["text"]).strip() if i - 1 in df.index else ""
            walter_line = str(df.iloc[i]["text"]).strip()
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_line}\n<|assistant|>\n{walter_line}"
            prompts.append(prompt)

        return Dataset.from_pandas(pd.DataFrame({"prompt": prompts}))

