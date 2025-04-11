from src.data.datamodules import BrainMRDataModule
import pandas as pd
from generative.networks.nets import AutoencoderKL
import torch
import numpy as np
from tqdm import tqdm
import argparse


def main(csv_path, ae_model_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dm = BrainMRDataModule(
        csv_path=csv_path,
        batch_size=1,
        num_workers=4,
        val_size=0.005,
        test_size=0.1,
        resize_to=[128, 160, 128],
        seed=42,
    )

    dm.setup('fit')
    dm.setup('test')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Get the df from csv
    df = pd.read_csv(csv_path)

    df["latent_path"] = None
    df["mode"] = None

    # Model
    print("Loading pretrained AutoencoderKL...")
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 128, 128),
        latent_channels=3,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )
    model.load_state_dict(torch.load(ae_model_path, map_location=device))
    model = model.to(device)

    for loader, mode in zip([train_loader, val_loader, test_loader], ['train', 'val', 'test']):
        print(f"Processing {mode} set...")
        for batch in tqdm(loader):
            x, path = batch["image"], batch["path"]
            x = x.to(device)
            model.eval()
            with torch.inference_mode():
                z = model.encode_stage_2_inputs(x)
                # Latent path
                destpath = path[0].replace('.nii.gz', '_compressed_latent.npz').replace('.nii', '_compressed_latent.npz')
                z = z.cpu().squeeze(0).numpy()
                np.savez_compressed(destpath, data=z)
                # Update the dataframe
                df.loc[df['path'] == path[0], 'latent_path'] = destpath
                df.loc[df['path'] == path[0], 'mode'] = mode

    print("Saving new dataframe...")
    df.to_csv(csv_path.replace('.csv', "_with_compressed_latent.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for saving compressed latents with the AutoencoderKL.")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("--ae_model_path", type=str, help="Path to the pretrained AutoencoderKL model")

    args = parser.parse_args()
    main(args.csv_path, args.ae_model_path)