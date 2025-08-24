import os
import gc
import pandas as pd
import torch
import huggingface_hub
from datasets import Dataset
import warnings

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
    GPU-Safe CharacterChatBot that gracefully handles CPU/GPU environments
    """

    def __init__(self, model_path: str, data_path: str = "/content/data/BB_data.csv",
                 huggingface_token: str | None = None, force_cpu: bool = False):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        
        # Determine device and model strategy
        self.has_gpu = torch.cuda.is_available() and not force_cpu
        self.device = "cuda" if self.has_gpu else "cpu"
        
        # Choose appropriate model based on hardware
        if self.has_gpu:
            self.base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            print("No GPU detected. Using smaller CPU-friendly model...")
            self.base_model_path = "microsoft/DialoGPT-medium"
        
        print(f"Device: {self.device}")
        print(f"Base model: {self.base_model_path}")
        
        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)

        # Clear any existing memory
        self._clear_memory()

        # Load or train model
        try:
            if huggingface_hub.repo_exists(self.model_path):
                print(f"Loading existing model from {self.model_path}")
                self.pipeline = self.load_model(self.model_path)
            else:
                print("Model not found. Training new model...")
                train_dataset = self.load_data()
                self.train(self.base_model_path, train_dataset)
                self.pipeline = self.load_model(self.model_path)
        except Exception as e:
            print(f"Error with custom model: {e}")
            print("Falling back to base model...")
            self.pipeline = self.load_model(self.base_model_path)

    def _clear_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass  # Ignore CUDA errors if no driver

    def _get_model_config(self):
        """Get model configuration based on available hardware"""
        if self.has_gpu:
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
        else:
            return {
                "torch_dtype": torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

    # ---------------------------
    # Chat interface
    # ---------------------------
    def chat(self, message: str, history: list[list[str]]):
        try:
            tokenizer = self.pipeline.tokenizer
            
            # Handle different model types
            if "DialoGPT" in self.base_model_path:
                # DialoGPT format
                conversation = ""
                for user_msg, assistant_msg in history[-3:]:  # Limit context
                    conversation += f"{user_msg}{tokenizer.eos_token}{assistant_msg}{tokenizer.eos_token}"
                conversation += f"Walter White: {message}{tokenizer.eos_token}"
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    outputs = self.pipeline(
                        conversation,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        return_full_text=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = outputs[0]["generated_text"]
                # Clean up response
                response = response.split(tokenizer.eos_token)[0].strip()
                return response
            
            else:
                # Llama format
                messages = [{"role": "system", "content": "You are Walter White from Breaking Bad. Respond in character."}]
                for user_msg, assistant_msg in history[-5:]:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                messages.append({"role": "user", "content": message})

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                terminators = [tokenizer.eos_token_id]
                if tokenizer.convert_tokens_to_ids("<|eot_id|>") != tokenizer.unk_token_id:
                    terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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
                
        except Exception as e:
            print(f"Chat error: {e}")
            return "I'm having trouble responding right now. Try again."

    # ---------------------------
    # Load model + adapter
    # ---------------------------
    def load_model(self, model_or_adapter_repo: str):
        self._clear_memory()
        
        try:
            model_config = self._get_model_config()
            
            print(f"Loading tokenizer from {self.base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"Loading base model: {self.base_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **model_config
            )

            # If loading an adapter
            if model_or_adapter_repo != self.base_model_path:
                try:
                    print(f"Loading LoRA adapter: {model_or_adapter_repo}")
                    model = PeftModel.from_pretrained(model, model_or_adapter_repo)
                except Exception as adapter_error:
                    print(f"Failed to load adapter: {adapter_error}")
                    # Try loading as full model
                    try:
                        print("Trying to load as full model...")
                        del model
                        self._clear_memory()
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_or_adapter_repo,
                            **model_config
                        )
                        tokenizer = AutoTokenizer.from_pretrained(model_or_adapter_repo, use_fast=True)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                    except Exception:
                        print("Failed to load custom model, using base model")
                        model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_path,
                            **model_config
                        )

            print("Creating pipeline...")
            pipeline_kwargs = {"torch_dtype": model_config["torch_dtype"]} if self.has_gpu else {}
            
            return pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs=pipeline_kwargs
            )
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise RuntimeError(f"Could not load model. Check GPU availability and model path. Error: {e}")

    # ---------------------------
    # Training function (GPU only)
    # ---------------------------
    def train(
        self,
        base_model_name_or_path: str,
        dataset: Dataset,
        output_dir: str = "./results",
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        optim: str = "paged_adamw_8bit",
        save_steps: int = 50,
        logging_steps: int = 10,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 0.3,
        max_steps: int = 100,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
    ):
        if not self.has_gpu:
            print("Training requires GPU. Skipping training and using base model.")
            return
            
        self._clear_memory()
        
        model_config = self._get_model_config()
        
        print("Loading model for training...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                **model_config
            )
            model.config.use_cache = False
        except Exception as e:
            print(f"Training setup failed: {e}")
            return

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=32,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
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
            packing=True,
            completion_only_loss=False,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            max_seq_length=512,
        )

        # Format dataset
        def format_dataset(examples):
            formatted_texts = []
            for p, c in zip(examples["prompt"], examples["completion"]):
                full_text = f"{p}{c}"
                # Truncate if too long
                if len(full_text.split()) > 400:
                    words = full_text.split()
                    full_text = " ".join(words[:400])
                formatted_texts.append(full_text)
            return {"text": formatted_texts}

        print("Formatting dataset...")
        formatted_dataset = dataset.map(
            format_dataset, 
            batched=True, 
            remove_columns=dataset.column_names
        )
        
        # Limit dataset size for stability
        if len(formatted_dataset) > 1000:
            formatted_dataset = formatted_dataset.select(range(1000))

        try:
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=formatted_dataset,
                peft_config=peft_config,
            )
            trainer.tokenizer = tokenizer

            print("Starting training...")
            trainer.train()

            # Save locally
            adapter_dir = "final_ckpt"
            trainer.model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

            # Clean up before pushing
            del trainer, model
            self._clear_memory()

            # Push to hub
            try:
                base_model_push = AutoModelForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    **model_config
                )
                peft_model = PeftModel.from_pretrained(base_model_push, adapter_dir)
                peft_model.push_to_hub(self.model_path)
                tokenizer.push_to_hub(self.model_path)
                
                del base_model_push, peft_model
                self._clear_memory()
                print("Successfully pushed to hub!")
            except Exception as e:
                print(f"Push to hub failed: {e}")
                print("Model saved locally in 'final_ckpt'")
                
        except Exception as e:
            print(f"Training failed: {e}")

    # ---------------------------
    # Data loading
    # ---------------------------
    def load_data(self) -> Dataset:
        print("Loading data...")
        df = pd.read_csv(self.data_path).dropna()
        df["text"] = df["text"].astype(str)
        df["number_of_words"] = df["text"].str.strip().str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        df["walter_response_flag"] = ((df["actor"] == "Walter") & (df["number_of_words"] > 5)).astype(int)

        idxs = df.index[df["walter_response_flag"] == 1].tolist()
        
        prompts, completions = [], []
        for i in idxs:
            user_line = str(df.iloc[i - 1]["text"]).strip() if i - 1 in df.index else ""
            walter_line = str(df.iloc[i]["text"]).strip()
            
            # Skip very long examples
            if len(user_line.split()) > 50 or len(walter_line.split()) > 50:
                continue
            
            if "DialoGPT" in self.base_model_path:
                prompt_text = f"User: {user_line}\nWalter White: "
            else:
                system_prompt = "You are Walter White from Breaking Bad. Respond in his style.\n"
                prompt_text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_line}\n<|assistant|>\n"
            
            prompts.append(prompt_text)
            completions.append(walter_line)

        print(f"Created {len(prompts)} training examples")
        return Dataset.from_pandas(pd.DataFrame({"prompt": prompts, "completion": completions}))


# Helper function for easy instantiation
def create_safe_chatbot(model_path: str, data_path: str = "/content/data/BB_data.csv", 
                       huggingface_token: str = None, force_cpu: bool = False):
    """Create a chatbot that works on both CPU and GPU"""
    return CharacterChatBot(
        model_path=model_path,
        data_path=data_path,
        huggingface_token=huggingface_token,
        force_cpu=force_cpu
    )
