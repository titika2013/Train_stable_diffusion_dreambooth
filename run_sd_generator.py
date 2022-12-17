import argparse
import os
import random
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, LMSDiscreteScheduler
from torch import autocast
from IPython.display import display

def prep_pipe(path_to_model, scheduler_name="DDIMScheduler"):
    if scheduler_name != "DPMSolverMultistepScheduler" or "DDIMScheduler" or "LMSDiscreteScheduler":
        scheduler_name = "DDIMScheduler"
    if scheduler_name == "DPMSolverMultistepScheduler":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(path_to_model, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            path_to_model,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float16,
            solver_order=2,
            clip_sample=False,
        ).to("cuda")
    elif scheduler_name == "DDIMScheduler":
        scheduler = DDIMScheduler.from_pretrained(path_to_model, subfolder="scheduler", clip_sample=False)
        # choose scheduler DPMSolverMultistepScheduler, DDIMScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            path_to_model,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float16,
            clip_sample=False,
        ).to("cuda")
    elif scheduler_name == "LMSDiscreteScheduler":
        scheduler = LMSDiscreteScheduler.from_pretrained(path_to_model, subfolder="scheduler", clip_sample=False)
        # choose scheduler DPMSolverMultistepScheduler, DDIMScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            path_to_model,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float16,
            clip_sample=False,
        ).to("cuda")

    return pipe


def execute_generation_sd(model_sd_path, key_name="", input_user_prompt="portrait masterpiece painting by vasnetsov",
                          path_to_dest="", negative_prompt="", num_samples=4, guidance_scale=8, num_inference_steps=110,
                          height=512, width=512, seed=0, eta=0, need_display=True, need_add_prompt=True):
    if not path_to_dest:
        path_to_dest = model_sd_path
        path_to_save_img = path_to_dest + "/gen_images"
    else:
        path_to_save_img = path_to_dest
    if need_add_prompt:
        main_prompt = input_user_prompt
        prompt = key_name + " " + main_prompt  # @param {type:"string"}
    else:
        prompt = input_user_prompt

    # if seed != 0:
    #     generator = torch.Generator("cuda").manual_seed(1024)
    # else:
    #     generator = 0

    pipe = prep_pipe(model_sd_path)
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=None,
#             eta=eta,
        ).images

    if not os.path.exists(path_to_save_img):
        os.mkdir(path_to_save_img)
    i = 0
    for img in images:
        im_rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{path_to_save_img}/{prompt[0:20]}_{i}_{random.randint(0, 1000)}.jpg",
                    np.asarray(im_rgb))  ### Change !!!!
        i += 1
        if need_display:
            display(img)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model_sd_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="where save generated images"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        required=False,
        help="Negative prompt",
    )
    parser.add_argument(
        "--input_user_prompt",
        type=str,
        default="portrait masterpiece painting by vasnetsov",
        help="The prompt or prompts to guide the image generation",
    )

    parser.add_argument(
        "--token_prompt",
        type=str,
        default=None,
        help="prompt with training token",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=int,
        default=8,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance. "
             "guidance_scale is defined as w of equation 2. of Imagen Paper. "
             "Guidance scale is enabled by setting guidance_scale > 1. "
             "Higher guidance scale encourages to generate images that are closely linked to the text prompt, "
             "usually at the expense of lower image quality, better from 5 to 14"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="The number of denoising steps. More denoising steps usually "
             "lead to a higher quality image at the expense of slower inference"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. "
             "Only applies to schedulers.DDIMScheduler, will be ignored for others."
    )
    parser.add_argument(
        "--need_display",
        type=bool,
        default=True,
        help="Need to display generated images"
    )
    parser.add_argument(
        "--need_add_key_prompt_to_input_user_prompt",
        type=bool,
        default=True,
        help="add token prompt to start input user prompt"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    execute_generation_sd(model_sd_path=args.model_sd_path, key_name=args.token_prompt,
                          input_user_prompt=args.input_user_prompt,
                          path_to_dest=args.save_dir, negative_prompt=args.negative_prompt,
                          num_samples=args.num_samples,
                          guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                          height=args.resolution, width=args.resolution, seed=args.seed, eta=args.eta,
                          need_display=args.need_display, need_add_prompt=args.need_add_key_prompt_to_input_user_prompt)


if __name__ == '__main__':
    args = parse_args()
    main(args)

    # save_model_new_place(session_name, prev_model_out_dir, path_to_dest="", main_dir=os.getcwd()):
