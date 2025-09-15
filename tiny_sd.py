import torch
from diffusers import DiffusionPipeline

from mycelya_torch import RemoteMachine

machine = RemoteMachine("mock")

pipeline = DiffusionPipeline.from_pretrained(
    "segmind/tiny-sd", torch_dtype=torch.float16
).to(machine.device("cpu"))
prompt = "Portrait of a pretty girl"
image = pipeline(prompt).images[0]
image.save("my_image.png")
