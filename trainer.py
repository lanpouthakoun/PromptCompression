# trainer.py
import torch
import torch.nn.functional as F
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
import numpy as np
import os
from datasets import load_dataset


from model import load_models


class GRPOTrainer:
    def __init__(
        self,
        model_name: str,
        frozen_model_name: str, 
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 128,
        output_dir: str = "./output",
        seed: int = 42,
    ):

        self.max_length = max_length
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.checkpoint_dir = os.path.join('/checkpoints', 'grpo_checkpoints')
        self.seed = seed
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_dir = os.path.join('/checkpoints', 'logs') 
        os.makedirs(self.log_dir, exist_ok=True)  
        path = os.path.join(self.checkpoint_dir, "best_model")
        self.model, self.frozen_model, self.tokenizer, self.frozen_tokenizer = load_models(
            model_name, frozen_model_name
        )
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_dataset = load_dataset("super_glue", "multirc", split="train")
        self.train_dataset = self.train_dataset.rename_column("paragraph", "prompt")
        self.eval_dataset = load_dataset("super_glue", "multirc", split="test")
        self.eval_dataset = self.eval_dataset.rename_column("paragraph", "prompt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.frozen_model = self.frozen_model.to(self.device)
        self.frozen_model.eval()
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        
        self.train_dataset = self.train_dataset.shuffle(seed=self.seed).select(range(min(70 * self.batch_size, len(self.train_dataset))))

        self.eval_dataset = self.eval_dataset.shuffle(seed=self.seed).select(range(min(70 *self.batch_size, len(self.eval_dataset))))

        self.output_dir = output_dir

        self.grpo_config = GRPOConfig(
            seed=self.seed,
            generation_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_dir =self.log_dir,
            output_dir = self.output_dir,
            logging_steps = 20,
            report_to=["tensorboard"],
            save_steps = 50,
            max_completion_length = 512,
        )


        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            reward_funcs = self.compute_reward_fn,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset = self.eval_dataset,
            args=self.grpo_config,
        )

    def compute_reward_fn(self, prompts, completions, question, answer, label, **kwargs):
        
        rewards = []
        batch_size = len(prompts)
        
        eval_prompts = []
        compression_ratios = []
        true_labels = []
        for i in range(batch_size):

            q = question[i] if isinstance(question, list) else question
            a = answer[i] if isinstance(answer, list) else answer
            l = label[i] if isinstance(label, list) else label

            original_length = len(prompts[i].split(" "))
            compressed_length = len(completions[i].split(" "))

            if original_length == 0:
                compression_ratio = 1.0
            else:
                compression_ratio = compressed_length / original_length
                compression_ratio = max(0.1, min(3.0, compression_ratio))
            
            compression_score = 1.0 - compression_ratio
            compression_ratios.append(compression_score)
            
            eval_prompt = f"""Based on the following passage, answer whether the given answer is correct.

                Begin Passage: {completions[i]} End Of Passage. 

                Question: {q}

                Proposed Answer: {a}

                Is this answer correct based on the passage? Respond with ONLY the word "yes" or "no". Answer:"""
            
            eval_prompts.append(eval_prompt)
            

            if isinstance(l, (int, float)):
                true_label_str = "yes" if int(l) == 1 else "no"
            else:
                true_label_str = str(l).lower()
            true_labels.append(true_label_str)
        
        with torch.no_grad():
            self.frozen_tokenizer.padding_side = "left"
            eval_inputs = self.frozen_tokenizer(
                eval_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding = True
                ).to(self.device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.frozen_model.generate(
                    **eval_inputs,
                    min_new_tokens = 1,
                    max_new_tokens=5,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.frozen_tokenizer.pad_token_id,
                )
            for i in range(batch_size):

                input_ids = eval_inputs["input_ids"][i]
                attention_mask = eval_inputs["attention_mask"][i]
                actual_input_length = attention_mask.sum().item()
                

                generated_tokens = outputs[i][actual_input_length:]

                response = self.frozen_tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip().lower()

                response_text = response.split()[0] if response else ""
                if response_text in ["yes", "yes.", "yes,"]:
                    predicted_answer = "yes"
                elif response_text in ["no", "no.", "no,"]:
                    predicted_answer = "no"
                else:
                    predicted_answer = "invalid"
                

                
                accuracy_reward = 0.0
                if predicted_answer == true_labels[i]:
                    accuracy_reward = 1.0      # Correct answer
                elif predicted_answer == "invalid":
                    accuracy_reward = -0.5     # Invalid response penalty
                else:
                    accuracy_reward = -0.2     # Wrong answer penalty

                total_reward = 0.6 * accuracy_reward + 0.4 * compression_ratios[i]
                
                total_reward += np.random.normal(0, 0.01)
                
                rewards.append(total_reward)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        return rewards
    
    def evaluate_random_samples(self, num_samples: int = 10):
        
        eval_indices = np.random.choice(len(self.eval_dataset), size=min(num_samples, len(self.eval_dataset)), replace=False)
        selected_samples = [self.eval_dataset[int(i)] for i in eval_indices]
        
        
        with torch.no_grad():
            self.model.eval()  
            
            for i, sample in enumerate(selected_samples):
                prompt = sample['prompt']
                print(f"Original Prompt: {prompt}")
                print(f"Original prompt length: {len(prompt.split())} words")
                
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(
                        **inputs,
                        min_new_tokens=10,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                print(f"Completion: {completion}")
                print(f"Generated completion length: {len(completion.split())} words")
                
            self.model.train()  
        

    def train(self):
        # self.evaluate_random_samples()
        self.grpo_trainer.train()   
        path = os.path.join(self.checkpoint_dir, "best_model")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path) 
        self.tokenizer.save_pretrained(path)
        
        print("start evaluating")
        eval_res = self.grpo_trainer.evaluate()
        print(eval_res)