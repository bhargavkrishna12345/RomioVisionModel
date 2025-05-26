# Complete script with proper image building order

from dataclasses import dataclass
from pathlib import Path
import modal

# App definition
app = modal.App(name="Romio-plush-marketing-campaign-with-Flux-LoRA-Dreambooth")

# Build the base image with all dependencies and git operations FIRST
GIT_SHA = "e649678bf55aeaa4b60bd1f68b1ee726278c0304"

base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate==0.31.0",
        "datasets~=2.13.0",
        "fastapi[standard]==0.115.4",
        "ftfy~=6.1.0",
        "gradio~=5.5.0",
        "huggingface-hub==0.26.2",
        "hf_transfer==0.1.8",
        "numpy<2",
        "peft==0.11.1",
        "pydantic==2.9.2",
        "sentencepiece>=0.1.91,!=0.1.92",
        "smart_open~=6.4.0",
        "starlette==0.41.2",
        "transformers~=4.41.2",
        "torch~=2.2.0",
        "torchvision~=0.16",
        "triton~=2.2.0",
        "wandb==0.17.6",
    )
    .apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# NOW add the local training images as the final step
image = base_image.add_local_dir("./training_images", remote_path="/training_images")

# Configuration classes
@dataclass
class SharedConfig:
    """Configuration information shared across project components."""
    instance_name: str = "Romio"
    class_name: str = " reindeer plush"
    model_name: str = "black-forest-labs/FLUX.1-dev"

@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""
    prefix: str = "a photo of"
    postfix: str = ""
    resolution: int = 512
    train_batch_size: int = 4
    rank: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 2000
    checkpointing_steps: int = 1000
    seed: int = 142

@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""
    num_inference_steps: int = 50
    guidance_scale: float = 6

# Volume and secrets
volume = modal.Volume.from_name("dreambooth-finetuning-volume-flux", create_if_missing=True)
MODEL_DIR = "/model"

huggingface_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
USE_WANDB = True

# Load local images function
def load_local_images() -> Path:
    import PIL.Image
    from pathlib import Path as LocalPath
    
    # Source directory (where images are mounted in container)
    source_dir = LocalPath("/training_images")
    
    # Destination directory (where training script expects them)
    img_path = Path("/img")
    img_path.mkdir(parents=True, exist_ok=True)
    
    # Get all JPEG files
    jpeg_files = (list(source_dir.glob("*.jpg")) + 
                  list(source_dir.glob("*.jpeg")) + 
                  list(source_dir.glob("*.JPG")) + 
                  list(source_dir.glob("*.JPEG")))
    
    if not jpeg_files:
        raise ValueError(f"No JPEG files found in {source_dir}")
    
    # Process and copy images
    for ii, img_file in enumerate(jpeg_files):
        with PIL.Image.open(img_file) as image:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(img_path / f"{ii}.jpeg", "JPEG")
    
    print(f"{len(jpeg_files)} images loaded from {source_dir}")
    return img_path

# Download models function
@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    secrets=[huggingface_secret],
    timeout=600,
)
def download_models(config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],
    )

    DiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)

# Training function
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_DIR: volume},
    timeout=1800,
    secrets=[huggingface_secret]
    + (
        [modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])]
        if USE_WANDB
        else []
    ),
)
def train(config):
    import subprocess
    from accelerate.utils import write_basic_config

    # Load local images
    img_path = load_local_images()

    # Set up accelerate
    write_basic_config(mixed_precision="bf16")

    # Define training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

    def _exec_subprocess(cmd: list[str]):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # Run training
    print("launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "/root/examples/dreambooth/train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",
            f"--pretrained_model_name_or_path={MODEL_DIR}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={MODEL_DIR}",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",
        ]
        + (
            [
                "--report_to=wandb",
            ]
            if USE_WANDB
            else []
        ),
    )
    volume.commit()

# Model inference class
@app.cls(image=image, gpu="A100", volumes={MODEL_DIR: volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        volume.reload()

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.load_lora_weights(MODEL_DIR)
        self.pipe = pipe

    @modal.method()
    def inference(self, text, config):
        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]
        return image

# Web interface (keeping original structure)
web_image = image

@app.function(image=web_image, max_containers=1)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()

    def go(text=""):
        if not text:
            text = example_prompts[0]
        return Model().inference.remote(text, config)

    config = AppConfig()
    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
    f"{instance_phrase} in cozy Christmas living room, warm fireplace lighting, holiday marketing campaign",
    f"commercial photography of {instance_phrase} as bedtime companion, soft bedroom lighting, lifestyle marketing shot",
    f"{instance_phrase} adventure scene in snowy forest, winter wonderland campaign, magical Christmas atmosphere",
    f"e-commerce product shot of {instance_phrase}, clean white background, professional studio lighting, online retail photography",
    f"social media content featuring {instance_phrase} with happy children playing, family lifestyle campaign, Instagram-ready composition",
    f"vintage Christmas poster style featuring {instance_phrase}, retro holiday advertising, nostalgic marketing design",
    f"packaging design concept with {instance_phrase} as hero product, colorful toy packaging, shelf-appeal marketing",
    f"emotional support campaign showing {instance_phrase} comforting child, heartwarming commercial photography style",
    f"seasonal gift guide featuring {instance_phrase} under Christmas tree, holiday shopping campaign, magazine layout style",
    f"brand mascot design of {instance_phrase} with logo placement, corporate identity campaign, professional branding photography"
                ]



    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration."""

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    with gr.Blocks(
        theme=theme,
        title=f"Generate images of {config.instance_name} on Modal",
    ) as interface:
        gr.Markdown(f"# Generate images of {instance_phrase}.\n\n{description}")
        
        with gr.Row():
            inp = gr.Textbox(
                label="",
                placeholder=f"Describe the version of {instance_phrase} you'd like to see",
                lines=10,
            )
            out = gr.Image(
                height=512, width=512, label="", min_width=512, elem_id="output"
            )
        
        with gr.Row():
            btn = gr.Button("Dream", variant="primary", scale=2)
            btn.click(fn=go, inputs=inp, outputs=out)

            gr.Button(
                "⚡️ Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        with gr.Column(variant="compact"):
            for ii, prompt in enumerate(example_prompts):
                btn = gr.Button(prompt, variant="secondary")
                btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

    return mount_gradio_app(app=web_app, blocks=interface, path="/")

# Entry point
@app.local_entrypoint()
def run(max_train_steps: int = 250):
    print("🎨 loading model")
    download_models.remote(SharedConfig())
    print("🎨 setting up training")
    config = TrainConfig(max_train_steps=max_train_steps)
    train.remote(config)
    print("🎨 training finished")