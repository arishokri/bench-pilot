from __future__ import annotations

import time
from dataclasses import dataclass

from benchpilot.benchmarks.base import BenchmarkResult
from benchpilot.benchmarks.gpu._torch import cuda_or_skip


@dataclass
class StableDiffusionInference:
    """Inference throughput on SDXL-Turbo at 1 step (fast & comparable).

    First run will download the model (~6GB). Subsequent runs use the HF cache.
    """

    name: str = "gpu.image_gen.sdxl_turbo"
    component: str = "gpu"
    num_images: int = 4
    height: int = 512
    width: int = 512
    steps: int = 1
    guidance: float = 0.0
    model_id: str = "stabilityai/sdxl-turbo"

    def params(self) -> dict:
        return {
            "num_images": self.num_images, "height": self.height, "width": self.width,
            "steps": self.steps, "guidance": self.guidance, "model_id": self.model_id,
        }

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})
        try:
            from diffusers import AutoPipelineForText2Image  # type: ignore
        except ImportError:
            return BenchmarkResult(results={"skipped": "diffusers not installed"})

        # Silence diffusers' own internal deprecation warning about `upcast_vae`.
        # The pipeline still uses it on SDXL VAE (force_upcast=True) until they
        # finish the 1.0 migration; doing the cast ourselves doesn't work because
        # the pipeline also converts latents inside the same code branch.
        import warnings
        warnings.filterwarnings("ignore", message=".*upcast_vae.*", category=FutureWarning)
        pipe = AutoPipelineForText2Image.from_pretrained(
            # diffusers 0.37 still uses `torch_dtype=`; transformers 5.x switched to `dtype=`.
            self.model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)

        # Warmup
        with torch.inference_mode():
            _ = pipe(prompt="a cat", num_inference_steps=1, guidance_scale=0.0,
                     height=self.height, width=self.width).images
        torch.cuda.synchronize()

        start = time.monotonic()
        with torch.inference_mode():
            imgs = pipe(
                prompt=["a fox", "a mountain", "a ship", "a cathedral"][: self.num_images],
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                height=self.height, width=self.width,
            ).images
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start

        n = len(imgs)
        del pipe, imgs
        torch.cuda.empty_cache()
        return BenchmarkResult(results={
            "images": n,
            "elapsed_s": elapsed,
            "seconds_per_image": elapsed / n,
            "images_per_second": n / elapsed,
        })
