import itertools
import os
import random
import shutil
from argparse import Namespace
from subprocess import getoutput

import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from image_crop import img_process
from my_utils import free_gpu_cache, training_function


####################################################################################################################
####################################################################################################################
####################################################################################################################


def run_training_model(folder_with_user_photo="", where_temp_save_res_photos="", output_dir="", class_dir="",
                       model_sd_path="",
                       key_name="TIG-XXL", need_face_find=False, main_dir="", image_resolution=512, seed=42,
                       save_model_checkpoint_every=5000):

    if not main_dir:
        main_dir = os.getcwd()
        # cur folder
    if not folder_with_user_photo:
        folder_with_user_photo = main_dir + "/drive/MyDrive/fresh_photos"  # @param{type: 'string'}
    if not key_name:
        key_name = key_name  # "TIG-XXL"
    # This type of name works better
    # https://www.arxiv-vanity.com/papers/2208.12242/
    if not where_temp_save_res_photos:
        where_temp_save_res_photos = main_dir + "/" + f"PHOTOS_{key_name}"

    img_process(source_photo_dir=folder_with_user_photo, dst_photo_dir=where_temp_save_res_photos, crop_size=512,
                face_detection=need_face_find, key_of_photo=key_name)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # Create/Load Session

    Session_Name = f"{key_name}_session"

    while Session_Name == "":
        print('[1;31mInput the Session Name:')
        Session_Name = input('')

    # pretrained = False
    # contains_faces = "Both"  # MAYBE MODIFY DEEPFACE GENDER

    if not model_sd_path:
        model_sd_path = main_dir + '/stable-diffusion-v1-5'
    if not output_dir:
        output_dir = main_dir + '/models/' + Session_Name  # !!!!!!!!!!!!!!!!! where to save the model


    pt = ""
    if not class_dir:
        class_dir = main_dir + '/Regularization_images'
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # create_images_folder(MAIN_DIR=MAIN_DIR, INSTANCE_DIR=INSTANCE_DIR, IMAGES_FOLDER=WHERE_SAVE_RES_PHOTOS,
    #                      Remove_existing_instance_images=True)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    training_steps = len(os.listdir(where_temp_save_res_photos)) * 200
    training_steps = training_steps if training_steps <= 3000 else 3000
    # Total Steps = Number of Instance images * 200, if you use 30 images, use 6000 steps, if you're not satisfied with
    # the result, resume training for another 500 steps, and so on ...

    if seed == '' or seed == '0':
        seed = random.randint(1, 999999)

    if not image_resolution:
        image_resolution = "512"  # ["512", "576", "640", "704", "768", "832", "896", "960", "1024"]
    res = int(image_resolution)

    fp16 = True
    if fp16:
        prec = "fp16"
    else:
        prec = "no"
    # Enable/disable half-precision, disabling it will double the training time and produce 4GB-5.2GB checkpoints.

    # GC= "--gradient_checkpointing"
    s = getoutput('nvidia-smi')
    if 'A100' in s:
        precision = "no"
        # GC= ""
    else:
        precision = prec

    train_text_encoder_for = 30
    if train_text_encoder_for >= 100:
        stptxt = training_steps
    elif train_text_encoder_for == 0:
        stptxt = 10
    else:
        stptxt = int((training_steps * train_text_encoder_for) / 130)

    save_checkpoint_every_n_steps = False
    if not save_model_checkpoint_every:
        save_model_checkpoint_every = 10000

    stp = 0

    start_saving_from_the_step = 500

    stpsv = start_saving_from_the_step

    if save_checkpoint_every_n_steps:
        stp = save_model_checkpoint_every
    # Start saving intermediary checkpoints from this step.

    # Disconnect_after_training=False

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # @title Load the Stable Diffusion model

    prior_preservation_class_prompt = ""

    text_encoder = CLIPTextModel.from_pretrained(
        model_sd_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        model_sd_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_sd_path, subfolder="unet"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_sd_path,
        subfolder="tokenizer",
    )

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    # @title Setting up all training args
    args = Namespace(
        pretrained_model_name_or_path=model_sd_path,
        resolution=res,
        center_crop=False,
        train_text_encoder=True,
        save_starting_step=stpsv,
        stop_text_encoder_training=stptxt,
        save_n_steps=stp,
        instance_data_dir=where_temp_save_res_photos,
        instance_prompt=pt,
        learning_rate=2e-6,
        max_train_steps=training_steps,
        train_batch_size=1,  # set to 1 if using prior preservation
        with_prior_preservation=False,
        dient_accumulation_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        mixed_precision=precision,
        gradient_checkpointing=True,  # set this to True to lower the memory usage.
        use_8bit_adam=True,  # use 8bit optimizer from bitsandbytes
        seed=seed,
        prior_loss_weight=0.1,
        sample_batch_size=1,
        class_prompt=prior_preservation_class_prompt,
        class_data_dir=class_dir,
        save_steps=save_model_checkpoint_every,
        lr_scheduler="polynomial",
        lr_warmup_steps=0,
        output_dir=output_dir,
    )

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    # @title Training function

    training_function(text_encoder, vae, unet, args, tokenizer)
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()

    shutil.move(where_temp_save_res_photos, output_dir + "/" + "USER_PHOTOS")
    with open(output_dir + "/Token_name.txt", 'w') as f:
        f.write(key_name)

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    free_gpu_cache()
