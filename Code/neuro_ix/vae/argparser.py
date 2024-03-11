import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--test", help="run with placeholder dataset", action="store_true"
    )
    parser.add_argument(
        "--project_dataset",
        help="project the dataset",
        action="store_true",
    )
    
    parser.add_argument(
        "-a", "--array_id", help="specify array job id", default=None, required=False
    )
    parser.add_argument(
        "-r",
        "--raw_images",
        help="if not test, use raw images of HCP instead of preprocessed brains",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--exclude_qc",
        help="should the dataset exclude images with QC errors",
        action="store_true",
    )

    parser.add_argument(
        "--max_epochs", default=100, required=False, help="Epochs to train", type=int
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        required=False,
        help="VAE learning rate",
        type=float,
    )
    parser.add_argument(
        "--beta", default=0.1, required=False, help="beta label loss weight", type=float
    )
    parser.add_argument(
        "--use_decoder",action="store_true"
    )
   
    parser.add_argument(
        "--batch_train",
        default=4,
        required=False,
        help="bacth size for train",
        type=int,
    )

    return parser.parse_args()
