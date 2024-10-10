import os,sys
import os.path as osp
try:
    comfy_utils = sys.modules["utils"]
except:
    print("have not find comfy utils module")

now_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(now_dir)

import time
import torch
import folder_paths
import numpy as np
from PIL import Image
from . import utils
sys.modules['utils'] = utils

from huggingface_hub import snapshot_download
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video

output_dir = folder_paths.get_output_directory()
pretrained_dir = osp.join(now_dir,"pretrained_models")


class PyramidFlowNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt":("TEXT",),
                "model_variant":(['768p','384p'],),
                "temp":([16,31],),
                "guidance_scale":("FLOAT",{
                    "default":9.0,
                    "min":7.0,
                    "max":9.0,
                    "step":0.1,
                    "round":0.01,
                    "display":"slider"
                }),
                "video_guidance_scale":("FLOAT",{
                    "default":5.0,
                })
            },
            "optional":{
                "image":("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_PyramidFlow"

    def gen_video(self,prompt,model_variant,temp,guidance_scale,
                  video_guidance_scale,image=None):
        if not osp.exists(osp.join(pretrained_dir,"diffusion_transformer_768p/diffusion_pytorch_model.bin")):
            snapshot_download(repo_id="rain1011/pyramid-flow-sd3",
                              local_dir=pretrained_dir,
                              local_dir_use_symlinks=False)
        torch.cuda.set_device(0)
        model_dtype, torch_dtype = 'bf16', torch.bfloat16
        model = PyramidDiTForVideoGeneration(
            pretrained_dir,
            model_dtype=model_dtype,
            model_variant='diffusion_transformer_'+model_variant
        )
        model.vae.to("cuda")
        model.dit.to("cuda")
        model.text_encoder.to("cuda")
        model.vae.enable_tiling()
        if model_variant == "768p":
            height,width = 768,1280
        else:
            height,width = 384,640
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            if image is None:
                frames = model.generate(
                    prompt=prompt,
                    num_inference_steps=[20, 20, 20],
                    video_num_inference_steps=[10, 10, 10],
                    height=height,     
                    width=width,
                    temp=temp,                    # temp=16: 5s, temp=31: 10s
                    guidance_scale=guidance_scale,         # The guidance for the first frame
                    video_guidance_scale=video_guidance_scale,   # The guidance for the other video latent
                    output_type="pil",
                    save_memory=True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
                )
            else:
                image_np = image.numpy()[0] * 255
                image_pil = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
                image_pil = image_pil.resize((width,height))
                frames = model.generate_i2v(
                    prompt=prompt,
                    input_image=image_pil,
                    num_inference_steps=[10, 10, 10],
                    temp=temp,
                    video_guidance_scale=video_guidance_scale,
                    output_type="pil",
                    save_memory=True,           # If you have enough GPU memory, set it to `False` to improve vae decoding speed
                )
        video_path = osp.join(output_dir,f"PyramidFlow_{time.time_ns()}.mp4")
        export_to_video(frames,video_path,fps=24)
        return (video_path,)

NODE_CLASS_MAPPINGS = {
    "PyramidFlowNode": PyramidFlowNode
}

sys.modules['utils'] = comfy_utils