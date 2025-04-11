import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import argparse
from src.ldae.utils import load_yaml_config
from src.ldae.nets import AttentionSemanticEncoder, ShiftUNet

def main(config_path, ckpt_path, output_folder):
    config = load_yaml_config(config_path)

    encoder_args = config['fit']['model']['init_args']['enc_args']
    decoder_args = config['fit']['model']['init_args']['unet_args']

    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoder_weights = {k.replace('ema_encoder.', ''): v for k, v in ckpt['state_dict'].items() if 'ema_encoder.' in k}
    decoder_weights = {k.replace('ema_decoder.', ''): v for k, v in ckpt['state_dict'].items() if 'ema_decoder.' in k}

    encoder = AttentionSemanticEncoder(**encoder_args)
    decoder = ShiftUNet(**decoder_args, latent_dim=encoder_args["emb_chans"])

    encoder.load_state_dict(encoder_weights, strict=False)
    decoder.load_state_dict(decoder_weights, strict=False)

    # Save encoder weights
    torch.save(encoder_weights, f'{output_folder}/encoder.pth')

    # Save decoder weights  
    torch.save(decoder_weights, f'{output_folder}/decoder.pth')

    print(f"Saved encoder and decoder weights to {output_folder}")


if __name__ == '__main__':
    sys.path.append('..')

    parser = argparse.ArgumentParser(description="Script for saving encoder and decoder weights from a trained model.")

    # Adding arguments
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_folder", type=str, default="models/", help="Output folder for saved weights")

    args = parser.parse_args()

    main(args.config, args.ckpt, args.output_folder)