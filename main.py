import modal
from pathlib import Path
import os

app = modal.App("ppo-training")

image = (
    modal.Image.debian_slim(python_version = "3.10")
    .pip_install(
        "torch==2.7.1",  
        "transformers",  
        "datasets",
        "accelerate",
        "peft",
        "tqdm",
        "numpy",
        "protobuf",
        "tensorboard",
        "tiktoken",
        "blobfile",
        "trl"
    )
    .add_local_dir(".", remote_path="/app")
)


volume = modal.Volume.from_name("compression-of-thought_trl_llama_to_olmo", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    gpu="h100",
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
    timeout=72000,
)
def train(
    model_name: str = "meta-llama/Llama-3.2-1B",
    frozen_model_name: str = "meta-llama/Llama-3.2-1B",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 128,
    output_dir: str = "/models/ppo_output",
    seed: int = 42,
):
    
    os.chdir("/app")
   
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    import sys
    sys.path.insert(0, "/app")
    from trainer import GRPOTrainer
    
    trainer = GRPOTrainer(
        model_name=model_name,
        frozen_model_name=frozen_model_name,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        output_dir=output_dir,
        seed=seed,
    )    
    
    trainer.train()
    
    
    volume.commit()
    
    return "done"

@app.local_entrypoint()
def main(
    model_name: str = "meta-llama/Llama-3.2-1B",
    frozen_model_name: str = "meta-llama/Llama-3.2-1B",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 128,
    output_dir: str = "/models/ppo_output",
    seed: int = 42,
):
    """Local entrypoint for Modal"""
    

    result = train.remote(
        model_name=model_name,
        frozen_model_name=frozen_model_name,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        output_dir=output_dir,
        seed=seed,
    )
    print(f"Training completed: {result}")


if __name__ == "__main__":
    main()