import os
import argparse

from run_training import run_training_model


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
        "--folder_with_user_photo",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--main_dir",
        type=str,
        default="",
        help="Current main directory if it will be set os.getcwd()",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--class_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the regularization data.",
    )
    parser.add_argument(
        "--key_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--save_model_checkpoint_every",
        type=int,
        default=5000,
        help="Checkpoint save every steps",
    )
    parser.add_argument(
        "--need_face_find",
        type=bool,
        default=False,
        help="Need find face on user images with deepface",
    )
    # parser.add_argument(
    #     "--save_n_steps",
    #     type=int,
    #     default=1,
    #     help=("Save the model every n global_steps"),
    # )
    #

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if not args.key_prompt:
        _, tail = os.path.split(args.folder_with_user_photo)
        args.key_prompt = tail

    # if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    return args


def main(args):
    run_training_model(model_sd_path=args.model_sd_path, folder_with_user_photo=args.folder_with_user_photo,
                       image_resolution=args.resolution, save_model_checkpoint_every=args.save_model_checkpoint_every,
                       key_name=args.key_prompt, output_dir=args.output_dir, need_face_find=args.need_face_find)


if __name__ == '__main__':
    args = parse_args()
    main(args)

    # save_model_new_place(session_name, prev_model_out_dir, path_to_dest="", main_dir=os.getcwd()):
