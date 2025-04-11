import re
import torch
import yaml
from tqdm import tqdm
from src.data.datamodules import BrainMRDataModule
from src.ldae.nets import AttentionSemanticEncoder
import argparse


def load_enc_args(config_path: str):
    """Load the YAML config file and return only the enc_args."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # Safely load the YAML into a Python dict

    # Extract the enc_args from the nested dictionary structure
    enc_args = config['fit']['model']['init_args']['enc_args']
    return enc_args


def main(csv_path, model_path, config_path, model_name, target_model, suffix):

    dm = BrainMRDataModule(
        csv_path=csv_path,
        batch_size=1,
        resize_to=(128, 160, 128),
        val_size=0.005,
        test_size=0.1,
        num_workers=0,
        seed=42,
        classes=['AD', 'CN', 'MCI'],
        load_images=True,
        load_latents=True,
        fake_3d=True,
        slicing_plane='axial'
    )

    dm.setup('fit')
    dm.setup('test')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    enc_args = load_enc_args(config_path)
    model = AttentionSemanticEncoder(**enc_args).to(device)
    # Load the weights of the model
    data = torch.load(model_path, map_location="cpu")
    weights = {key.replace(f'{target_model}.', ''): value for key, value in data['state_dict'].items()}
    model.load_state_dict(weights, strict=False)
    print("Successfully loaded the pretrained encoder model.")

    # Prepare containers for training data
    embeddings = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    ages = torch.tensor([]).to(device)

    # NEW: lists for string/int data
    subject_ids = []
    paths = []
    session_numbers = []

    model.eval()
    print(f"Saving embeddings and labels for the training set")
    for batch in tqdm(dm.train_dataloader()):
        # batch is something like:
        # {
        #   'image': <tensor>,
        #   'label': <tensor>,
        #   'age': <tensor>,
        #   'subject_id': [<string>...],
        #   'path': [<string>...],
        #   ...
        # }
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        age = batch['age'].to(device)

        # Strings are in python lists, one entry per sample
        batch_subject_ids = batch['subject_id']
        batch_paths = batch['path']

        with torch.no_grad():
            style_emb = model(image)

        # Accumulate embeddings and labels
        embeddings = torch.cat([embeddings, style_emb], dim=0)
        labels = torch.cat([labels, label], dim=0)
        ages = torch.cat([ages, age], dim=0)

        # For each entry, parse subject_id, path, session number
        # Since batch_size=1 in your code, these lists will have length 1
        for pth, sub_id in zip(batch_paths, batch_subject_ids):
            # Extract session number using a regex (ses-Mxxx -> xxx)
            match = re.search(r"ses-M(\d+)", pth)
            if match:
                session_num = int(match.group(1))
            else:
                session_num = None  # or -1, or handle error
            print(f"Subject ID: {sub_id}, Path: {pth}, Session Number: {session_num}")

            subject_ids.append(sub_id)
            paths.append(pth)
            session_numbers.append(session_num)

    # Save training set
    torch.save({
        'embeddings': embeddings,
        'labels': labels,
        'age': ages,
        'subject_id': subject_ids,
        'path': paths,
        'session_number': session_numbers
    }, f'{model_name}_train_embeddings_and_labels_{suffix}_{target_model}.pt')

    # Prepare containers for testing data
    test_embeddings = torch.tensor([]).to(device)
    test_labels = torch.tensor([]).to(device)
    test_ages = torch.tensor([]).to(device)

    # NEW: lists for string/int data in test
    test_subject_ids = []
    test_paths = []
    test_session_numbers = []

    print(f"Saving embeddings and labels for the test set")
    for batch in tqdm(dm.test_dataloader()):
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        age = batch['age'].to(device)

        batch_subject_ids = batch['subject_id']
        batch_paths = batch['path']

        with torch.no_grad():
            style_emb = model(image)

        # Accumulate embeddings and labels
        test_embeddings = torch.cat([test_embeddings, style_emb], dim=0)
        test_labels = torch.cat([test_labels, label], dim=0)
        test_ages = torch.cat([test_ages, age], dim=0)

        # For each entry, parse subject_id, path, session number
        for pth, sub_id in zip(batch_paths, batch_subject_ids):
            match = re.search(r"ses-M(\d+)", pth)
            if match:
                session_num = int(match.group(1))
            else:
                session_num = None

            test_subject_ids.append(sub_id)
            test_paths.append(pth)
            test_session_numbers.append(session_num)

    # Save test set
    torch.save({
        'embeddings': test_embeddings,
        'labels': test_labels,
        'age': test_ages,
        'subject_id': test_subject_ids,
        'path': test_paths,
        'session_number': test_session_numbers
    }, f'{model_name}_test_embeddings_and_labels_{suffix}_{target_model}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for extract semantic latent from the images given a pretrained LDAE.")
    # Adding arguments
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--target_model", type=str, required=True, default="ema_encoder", help="Target model name")
    parser.add_argument("--suffix", type=str, default='', help="Suffix for the saved files")
    args = parser.parse_args()
    # Call the main function with the parsed arguments
    main(args.csv_path, args.model_path, args.config_path, args.model_name, args.target_model, args.suffix)