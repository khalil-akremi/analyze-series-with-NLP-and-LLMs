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
    Optimized CharacterChatBot for Google Colab with memory management
    """

    def __init__(self, model_path: str, data_path: str = "/content/data/BB_data.csv",
                 huggingface_token: str | None = None, use_smaller_model: bool = False):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        
        # Use smaller model for Colab if needed
        if use_smaller_model:
            self.base_model_path = "microsoft/DialoGPT-medium"  # Much smaller alternative
        else:
            self.base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check GPU memory before proceeding
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available GPU memory: {gpu_memory:.1f} GB")
            if gpu_memory < 12:  # Less than 12GB
                print("Warning: Limited GPU memory detected. Consider using smaller model.")

        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)

        # Clear any existing GPU memory
        self._clear_memory()

        # If an adapter/model with this repo id already exists, load it; else train and push
        if huggingface_hub.repo_exists(self.model_path):
            print(f"Found existing model at {self.model_path}. Loading...")
            self.pipeline = self.load_model(self.model_path)
        else:
            print("Model not found in Hugging Face Hub. Training a new model...")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.pipeline = self.load_model(self.model_path)

    def _clear_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ---------------------------
    # Chat interface
    # ---------------------------
    def chat(self, message: str, history: list[list[str]]):
        tokenizer = self.pipeline.tokenizer
        
        # For DialoGPT, use simpler format
        if "DialoGPT" in self.base_model_path:
            # Simple concatenation for DialoGPT
            conversation = ""
            for user_msg, assistant_msg in history[-3:]:  # Keep only last 3 exchanges
                conversation += f"{user_msg}{tokenizer.eos_token}{assistant_msg}{tokenizer.eos_token}"
            conversation += message + tokenizer.eos_token
            
            outputs = self.pipeline(
                conversation,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            return outputs[0]["generated_text"].split(tokenizer.eos_token)[0].strip()
        
        # For Llama models
        else:
            messages = [{"role": "system",
                        "content": "You are Walter White from Breaking Bad. Respond in character.\n"}]
            # Limit history to prevent context overflow
            for user_msg, assistant_msg in history[-5:]:
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
        self._clear_memory()
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available! Falling back to CPU (very slow for large models)")
            if "Llama" in self.base_model_path:
                raise RuntimeError(
                    "Llama-3-8B requires GPU. Please enable GPU runtime in Colab or use a smaller model."
                )
        
        # More aggressive quantization for Colab (only if CUDA available)
        bnb_config = None
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Additional memory savings
            )

        # Handle different model architectures
        if "DialoGPT" in self.base_model_path:
            # DialoGPT doesn't need quantization (much smaller)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, use_fast=True)
            # Load with more memory-efficient settings
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if torch.cuda.is_available():
                model_kwargs.update({
                    "quantization_config": bnb_config,
                    "device_map": "auto",
                    "max_memory": {0: "10GB"},  # Limit GPU memory usage
                })
            
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_kwargs
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # If a LoRA adapter repo is provided (and not just the base), load it on top
        if model_or_adapter_repo != self.base_model_path:
            try:
                model = PeftModel.from_pretrained(model, model_or_adapter_repo)
                print(f"Successfully loaded LoRA adapter from {model_or_adapter_repo}")
            except Exception as e:
                print(f"Failed to load LoRA adapter: {e}")
                print("Attempting to load as full model...")
                # Cleanup first
                del model
                self._clear_memory()
                
                # Try loading as full model
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_or_adapter_repo,
                        quantization_config=bnb_config if "DialoGPT" not in model_or_adapter_repo else None,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_repo, use_fast=True)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as e2:
                    print(f"Failed to load model: {e2}")
                    raise e2

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.float16}
        )

    # ---------------------------
    # Training function with Colab optimizations
    # ---------------------------
    def train(
        self,
        base_model_name_or_path: str,
        dataset: Dataset,
        output_dir: str = "./results",
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,  # Increased for effective batch size
        optim: str = "paged_adamw_8bit",  # More memory efficient
        save_steps: int = 50,  # Save more frequently
        logging_steps: int = 5,
        learning_rate: float = 1e-4,  # Slightly lower
        max_grad_norm: float = 0.3,
        max_steps: int = 100,  # Reduced for faster training
        warmup_ratio: float = 0.1,  # Reduced warmup
        lr_scheduler_type: str = "cosine",
    ):
        self._clear_memory()
        
        # More aggressive quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading base model for training...")
        try:
            if "DialoGPT" in base_model_name_or_path:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "10GB"},
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This might be due to insufficient GPU memory or network issues.")
            raise e
            
        # Needed for gradient checkpointing + QLoRA training
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Smaller LoRA config for memory efficiency
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,  # Reduced dropout
            r=32,  # Reduced rank
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],  # Target fewer modules
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
            packing=True,  # Enable packing for efficiency
            completion_only_loss=False,
            dataloader_pin_memory=False,  # Save memory
            remove_unused_columns=True,
            max_seq_length=512,  # Limit sequence length
        )

        # Format dataset
        def format_dataset(examples):
            formatted_texts = []
            for p, c in zip(examples["prompt"], examples["completion"]):
                # Truncate if too long
                full_text = f"{p}{c}"
                if len(full_text.split()) > 400:  # Rough token limit
                    words = full_text.split()
                    full_text = " ".join(words[:400])
                formatted_texts.append(full_text)
            return {"text": formatted_texts}

        print("Formatting dataset...")
        formatted_dataset = dataset.map(
            format_dataset, 
            batched=True, 
            remove_columns=dataset.column_names,
            desc="Formatting dataset"
        )

        # Limit dataset size for Colab
        if len(formatted_dataset) > 1000:
            print(f"Limiting dataset from {len(formatted_dataset)} to 1000 examples for Colab")
            formatted_dataset = formatted_dataset.select(range(1000))

        print("Initializing trainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
        )

        trainer.tokenizer = tokenizer

        print("Starting training...")
        try:
            trainer.train()
        except Exception as e:
            print(f"Training error: {e}")
            # Save whatever we have
            adapter_dir = "emergency_ckpt"
            if hasattr(trainer, 'model'):
                trainer.model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
            raise e

        adapter_dir = "final_ckpt"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Clear memory before pushing
        del trainer, model
        self._clear_memory()

        print("Pushing to hub...")
        try:
            # Reload and push
            base_model_push = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                quantization_config=bnb_config if "DialoGPT" not in base_model_name_or_path else None,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            peft_model = PeftModel.from_pretrained(base_model_push, adapter_dir)
            peft_model.push_to_hub(self.model_path)
            tokenizer.push_to_hub(self.model_path)

            del base_model_push, peft_model
            self._clear_memory()
            print("Successfully pushed to hub!")
            
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("Model saved locally in 'final_ckpt' directory")

    # ---------------------------
    # Data loading
    # ---------------------------
    def load_data(self) -> Dataset:
        print("Loading and processing data...")
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
            
            # Skip if too long
            if len(user_line.split()) > 50 or len(walter_line.split()) > 50:
                continue
                
            if "DialoGPT" in self.base_model_path:
                # Simpler format for DialoGPT
                prompt_text = f"Walter White: {user_line} -> "
            else:
                prompt_text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_line}\n<|assistant|>\n"
            
            prompts.append(prompt_text)
            completions.append(walter_line)

        print(f"Created {len(prompts)} training examples")
        dataset = Dataset.from_pandas(pd.DataFrame({"prompt": prompts, "completion": completions}))
        return dataset


# Helper function to create chatbot with Colab-friendly settings
def create_colab_chatbot(model_path: str, data_path: str = "/content/data/BB_data.csv", 
                        huggingface_token: str = None, use_smaller_model: bool = True):
    """
    Create a chatbot optimized for Google Colab
    
    Args:
        model_path: HuggingFace model repository path
        data_path: Path to training data CSV
        huggingface_token: HuggingFace token for authentication
        use_smaller_model: Use DialoGPT instead of Llama for memory efficiency
    """
    return CharacterChatBot(
        model_path=model_path,
        data_path=data_path,
        huggingface_token=huggingface_token,
        use_smaller_model=use_smaller_model
    )


