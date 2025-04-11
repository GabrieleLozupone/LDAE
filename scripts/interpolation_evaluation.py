import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from generative.metrics import SSIMMetric


from src.data.datamodules.brain_mr_dm import BrainMRDataModule
from src.ldae.diffusion.gaussian_diffusion import GaussianDiffusion
from src.ldae import LatentDiffusionAutoencoders


def load_model(model_path, device):
    """Load and initialize the model from a checkpoint."""
    model = LatentDiffusionAutoencoders.load_from_checkpoint(model_path, map_location="cpu")
    data = torch.load(model_path, map_location="cpu")

    encoder_weights = {k.replace('ema_encoder.', ''): v for k, v in data['state_dict'].items() if 'ema_encoder.' in k}
    decoder_weights = {k.replace('ema_decoder.', ''): v for k, v in data['state_dict'].items() if 'ema_decoder.' in k}

    print("Loading the encoder and decoder weights.....")
    model.ema_encoder.load_state_dict(encoder_weights, strict=False)
    model.ema_decoder.load_state_dict(decoder_weights, strict=False)

    print("Moving the model to the device.....")
    model.vae = model.vae.to(device)
    model.ema_encoder = model.ema_encoder.to(device)
    model.ema_decoder = model.ema_decoder.to(device)

    return model


# Reuse your existing interpolation code
def lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b


def slerp(a, b, alpha):
    theta = calculate_theta(a, b)
    sin_theta = torch.sin(theta)
    return a * torch.sin((1.0 - alpha) * theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta


def calculate_theta(a, b):
    return torch.arccos(torch.dot(a.view(-1), b.view(-1)) / (torch.norm(a) * torch.norm(b)))


class InterpolationEvaluator:
    def __init__(self, model, dataset, device, debug=False, limit_samples=200):
        """
        Initialize the evaluator with the model and dataset

        Args:
            model: Your VAE model for encoding/decoding images
            dataset: Dataset containing the MRI scans
            device: Device to run computations on (cuda/cpu)
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.results = []
        self.debug = debug
        self.ssim = SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4)
        self.limit_samples = limit_samples

    def extract_session_number(self, path):
        """Extract session number from path string"""
        match = re.search(r"ses-M(\d+)", path)
        if match:
            return int(match.group(1))
        return None

    def organize_subject_data(self):
        """
        Organize dataset by subject_id and session number

        Returns:
            dict: Dictionary mapping subject_id to a dict of {session_num: dataset_index}
        """
        print("Organizing subject data...")
        subject_data = defaultdict(dict)

        if self.limit_samples:
            num_samples = self.limit_samples
        else:
            num_samples = len(self.dataset)

        # Iterate through dataset to find all subjects and their sessions
        for idx in tqdm(range(num_samples), desc="Organizing data"):
            sample = self.dataset[idx]
            subject_id = sample.get('subject_id')
            if isinstance(subject_id, list):  # Handle case where subject_id is a list
                subject_id = subject_id[0]

            path = sample.get('path')
            if isinstance(path, list):  # Handle case where path is a list
                path = path[0]

            session_num = self.extract_session_number(path)

            if subject_id is not None and session_num is not None:
                subject_data[subject_id][session_num] = idx
            if idx < 5:
                print(f"Subject ID: {subject_id}, Session: {session_num}, Index: {idx}")
                print(f"Subject data: {subject_data[subject_id][session_num]}")

        return subject_data

    def get_interpolation_pairs(self, subject_data):
        """
        Generate all possible interpolation pairs for each subject

        Args:
            subject_data: Dictionary mapping subjects to their session data

        Returns:
            list: List of dictionaries containing interpolation configurations
        """
        print("Generating interpolation pairs...")
        interpolation_configs = []

        for subject_id, sessions in subject_data.items():
            session_numbers = sorted(sessions.keys())

            if len(session_numbers) < 3:
                print(f"Skipping subject {subject_id} with only {len(session_numbers)} sessions - minimum 3 required")
                continue

            # For every pair of sessions (start, end)
            for i, start_session in enumerate(session_numbers):
                for j in range(i + 2, len(session_numbers)):  # Skip adjacent sessions
                    end_session = session_numbers[j]
                    start_idx = sessions[start_session]
                    end_idx = sessions[end_session]

                    # Find all sessions between start and end
                    for k in range(i + 1, j):
                        target_session = session_numbers[k]
                        target_idx = sessions[target_session]

                        # Calculate the interpolation alpha based on session time
                        alpha = (target_session - start_session) / (end_session - start_session)

                        interpolation_configs.append({
                            'subject_id': subject_id,
                            'start_session': start_session,
                            'end_session': end_session,
                            'target_session': target_session,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'target_idx': target_idx,
                            'alpha': alpha,
                            'time_gap': end_session - start_session
                        })
                        if self.debug:
                            print(f"\nInterpolation config: {interpolation_configs[-1]}\n")

        return interpolation_configs

    def perform_interpolation(self, config):
        """
        Perform interpolation for a given configuration

        Args:
            config: Dictionary containing interpolation configuration

        Returns:
            tuple: (generated_image, target_image)
        """
        print(f"Performing interpolation for config: {config}")

        # Get images from dataset
        start_sample = self.dataset[config['start_idx']]
        end_sample = self.dataset[config['end_idx']]
        target_sample = self.dataset[config['target_idx']]

        # Extract images and move to device
        image1 = start_sample["image"].unsqueeze(0).to(self.device)
        image2 = end_sample["image"].unsqueeze(0).to(self.device)
        target_image = target_sample["image"].unsqueeze(0).to(self.device)

        # Get latents if available, otherwise encode the images
        if 'latent' in start_sample and 'latent' in end_sample:
            latent1 = start_sample["latent"].unsqueeze(0).to(self.device)
            latent2 = end_sample["latent"].unsqueeze(0).to(self.device)
            latent_target = target_sample["latent"].unsqueeze(0).to(self.device)
        else:
            # Encode images if latents not available (would need to implement this)
            with torch.inference_mode():
                latent1 = self.model.vae(image1)
                latent2 = self.model.vae(image2)
                latent_target = self.model.vae(target_image)

        alpha = torch.tensor([config['alpha']]).to(self.device)

        if self.debug:
            x_interp = target_image
        else:
            with torch.inference_mode():
                # Scale latents as in your original code
                z1 = latent1 * self.model.scale_factor
                z2 = latent2 * self.model.scale_factor
                latent_target = latent_target * self.model.scale_factor

                # Encode the images to get semantic representation
                y_sem_1 = self.model.ema_encoder(image1)
                y_sem_2 = self.model.ema_encoder(image2)

                # Convert to latent space with diffusion model
                z_T_1 = self.model.gaussian_diffusion.latent_representation_learning_ddim_encode(
                    ddim_style='ddim100',
                    encoder=None,
                    decoder=self.model.ema_decoder,
                    x_0=None,
                    z_0=z1,
                    style=y_sem_1
                )

                z_T_2 = self.model.gaussian_diffusion.latent_representation_learning_ddim_encode(
                    ddim_style='ddim100',
                    encoder=None,
                    decoder=self.model.ema_decoder,
                    x_0=None,
                    z_0=z2,
                    style=y_sem_2
                )

                # Apply interpolation
                y_sem_interp = lerp(y_sem_1, y_sem_2, alpha)
                z_T_interp = slerp(z_T_1, z_T_2, alpha)

                # Generate interpolated image
                z_interp = self.model.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style='ddim100',
                    encoder=None,
                    decoder=self.model.ema_decoder,
                    x_0=None,
                    x_T=z_T_interp,
                    z=y_sem_interp,
                    disable_tqdm=True
                )

                x_interp = self.model.decode(z_interp)
                target_image = self.model.decode(latent_target)

        return x_interp, target_image

    def calculate_metrics(self, generated, target):
        """
        Calculate SSIM and MSE between generated and target images

        Args:
            generated: Generated image tensor
            target: Target image tensor

        Returns:
            dict: Dictionary containing SSIM and MSE values
        """
        ssim_score = self.ssim(generated, target)
        mse = torch.nn.functional.mse_loss(generated, target)

        # If debug return a random value for SSIM between 0 and 1 and MSE = 1 - the value obtained for SSIM
        if self.debug:
            return {
                'mse': np.random.rand(),
                'ssim': 1 - np.random.rand()
            }
        else:
            return {
                'mse': float(mse.mean()),
                'ssim': float(ssim_score.mean())
            }

    def evaluate_all(self):
        """
        Evaluate interpolation accuracy for all possible configurations

        Returns:
            DataFrame: Results containing metrics for each interpolation
        """
        subject_data = self.organize_subject_data()
        configs = self.get_interpolation_pairs(subject_data)

        for config in tqdm(configs, desc="Evaluating interpolations"):

            generated, target = self.perform_interpolation(config)
            metrics = self.calculate_metrics(generated, target)

            result = {
                'subject_id': config['subject_id'],
                'start_session': config['start_session'],
                'end_session': config['end_session'],
                'target_session': config['target_session'],
                'alpha': config['alpha'],
                'time_gap': config['time_gap'],
                'prediction_gap': min(config['target_session'] - config['start_session'],
                                      config['end_session'] - config['target_session']),
                'ssim': metrics['ssim'],
                'mse': metrics['mse']
            }

            print(f"Interpolation result: {result}")

            self.results.append(result)

        return pd.DataFrame(self.results)

    def analyze_results(self, results_df):
        """
        Analyze results to understand how interpolation accuracy varies with time gap

        Args:
            results_df: DataFrame containing evaluation results

        Returns:
            dict: Analysis results
        """
        # Add a column for relative position (how far into the gap is the target)
        results_df['relative_position'] = results_df.apply(
            lambda row: (row['target_session'] - row['start_session']) / row['time_gap'],
            axis=1
        )

        # Get unique time gaps
        time_gaps = sorted(results_df['time_gap'].unique())

        # Calculate statistics for each time gap
        gap_stats = results_df.groupby('time_gap').agg({
            'ssim': ['mean', 'std', 'min', 'max', 'count'],
            'mse': ['mean', 'std', 'min', 'max', 'count']
        })

        # Calculate statistics for each prediction gap
        prediction_gap_stats = results_df.groupby('prediction_gap').agg({
            'ssim': ['mean', 'std', 'min', 'max', 'count'],
            'mse': ['mean', 'std', 'min', 'max', 'count']
        })

        # Calculate statistics for each relative position (rounded to nearest 0.1)
        results_df['position_bucket'] = (results_df['relative_position'] * 10).round() / 10
        position_stats = results_df.groupby('position_bucket').agg({
            'ssim': ['mean', 'std', 'count'],
            'mse': ['mean', 'std', 'count']
        })

        return {
            'results_df': results_df,
            'gap_stats': gap_stats,
            'prediction_gap_stats': prediction_gap_stats,
            'position_stats': position_stats
        }

    def visualize_results(self, analysis_results):
        """
        Create visualizations of the analysis results

        Args:
            analysis_results: Results from analyze_results method

        Returns:
            dict: Dictionary containing matplotlib figures
        """
        results_df = analysis_results['results_df']
        gap_stats = analysis_results['gap_stats']
        prediction_gap_stats = analysis_results['prediction_gap_stats']
        position_stats = analysis_results['position_stats']

        figures = {}

        # Figure 1: SSIM vs Time Gap
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=results_df, x='time_gap', y='ssim', alpha=0.6, ax=ax1)

        # Add mean and std dev error bars
        time_gaps = sorted(gap_stats.index.tolist())
        mean_ssims = gap_stats[('ssim', 'mean')].tolist()
        std_ssims = gap_stats[('ssim', 'std')].tolist()

        ax1.errorbar(time_gaps, mean_ssims, yerr=std_ssims, fmt='o', color='red',
                     capsize=5, ecolor='black', markersize=8, label='Mean ± Std Dev')

        ax1.set_title('SSIM vs Time Gap')
        ax1.set_xlabel('Time Gap (months)')
        ax1.set_ylabel('SSIM (higher is better)')
        ax1.legend()
        figures['ssim_vs_gap'] = fig1

        # Figure 2: MSE vs Time Gap
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=results_df, x='time_gap', y='mse', alpha=0.6, ax=ax2)

        # Add mean and std dev error bars
        mean_mses = gap_stats[('mse', 'mean')].tolist()
        std_mses = gap_stats[('mse', 'std')].tolist()

        ax2.errorbar(time_gaps, mean_mses, yerr=std_mses, fmt='o', color='red',
                     capsize=5, ecolor='black', markersize=8, label='Mean ± Std Dev')

        ax2.set_title('MSE vs Time Gap')
        ax2.set_xlabel('Time Gap (months)')
        ax2.set_ylabel('MSE (lower is better)')
        ax2.legend()
        figures['mse_vs_gap'] = fig2

        # Figure 3: SSIM vs Prediction Gap
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        # Prepare data for plotting
        prediction_gaps = sorted(prediction_gap_stats.index.tolist())
        pred_mean_ssims = prediction_gap_stats[('ssim', 'mean')].tolist()
        pred_std_ssims = prediction_gap_stats[('ssim', 'std')].tolist()

        # Create bar plot with error bars
        ax3.bar(prediction_gaps, pred_mean_ssims, yerr=pred_std_ssims,
                capsize=5, color='skyblue', edgecolor='black', alpha=0.7)

        ax3.set_title('SSIM vs Prediction Gap')
        ax3.set_xlabel('Prediction Gap (months)')
        ax3.set_ylabel('Mean SSIM')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        figures['ssim_vs_prediction_gap'] = fig3

        # Figure 4: MSE vs Prediction Gap
        fig4, ax4 = plt.subplots(figsize=(12, 6))

        # Prepare data for plotting
        pred_mean_mses = prediction_gap_stats[('mse', 'mean')].tolist()
        pred_std_mses = prediction_gap_stats[('mse', 'std')].tolist()

        # Create bar plot with error bars
        ax4.bar(prediction_gaps, pred_mean_mses, yerr=pred_std_mses,
                capsize=5, color='salmon', edgecolor='black', alpha=0.7)

        ax4.set_title('MSE vs Prediction Gap')
        ax4.set_xlabel('Prediction Gap (months)')
        ax4.set_ylabel('Mean MSE')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        figures['mse_vs_prediction_gap'] = fig4

        # Figure 5: SSIM vs Relative Position
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=results_df, x='relative_position', y='ssim',
                        hue='time_gap', palette='viridis', ax=ax5)

        ax5.set_title('SSIM vs Relative Position')
        ax5.set_xlabel('Relative Position (0=start, 1=end)')
        ax5.set_ylabel('SSIM (higher is better)')
        figures['ssim_vs_relative_position'] = fig5

        # Figure 6: Mean SSIM by Position Bucket
        fig6, ax6 = plt.subplots(figsize=(12, 6))

        # Prepare data for plotting
        position_buckets = sorted(position_stats.index.tolist())
        pos_mean_ssims = position_stats[('ssim', 'mean')].tolist()
        pos_std_ssims = position_stats[('ssim', 'std')].tolist()

        # Create line plot with error region
        ax6.plot(position_buckets, pos_mean_ssims, 'o-', color='blue', linewidth=2, markersize=8)
        ax6.fill_between(position_buckets,
                         [pos_mean_ssims[i] - pos_std_ssims[i] for i in range(len(pos_mean_ssims))],
                         [pos_mean_ssims[i] + pos_std_ssims[i] for i in range(len(pos_mean_ssims))],
                         alpha=0.2, color='blue')

        ax6.set_title('Mean SSIM by Relative Position')
        ax6.set_xlabel('Relative Position (0=start, 1=end)')
        ax6.set_ylabel('Mean SSIM')
        ax6.grid(True, linestyle='--', alpha=0.7)
        figures['mean_ssim_by_position'] = fig6

        return figures

    def save_results(self, results_df, output_dir='interpolation_results'):
        """
        Save results to CSV and visualizations to files

        Args:
            results_df: DataFrame containing results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save results to CSV
        results_df.to_csv(os.path.join(output_dir, 'interpolation_results.csv'), index=False)

        # Create analysis and visualizations
        analysis = self.analyze_results(results_df)
        figures = self.visualize_results(analysis)

        # Save figures
        for name, fig in figures.items():
            fig.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')

        # Save time gap statistics
        analysis['gap_stats'].to_csv(os.path.join(output_dir, 'time_gap_statistics.csv'))

        # Save prediction gap statistics
        analysis['prediction_gap_stats'].to_csv(os.path.join(output_dir, 'prediction_gap_statistics.csv'))

        # Save position statistics
        analysis['position_stats'].to_csv(os.path.join(output_dir, 'position_statistics.csv'))

        # Create a summary report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Interpolation Evaluation Summary\n")
            f.write("===============================\n\n")

            f.write(f"Total interpolations evaluated: {len(results_df)}\n")
            f.write(f"Unique subjects: {results_df['subject_id'].nunique()}\n")
            f.write(f"Unique time gaps: {len(analysis['gap_stats'])}\n\n")

            f.write("Time Gap Statistics:\n")
            for gap, row in analysis['gap_stats'].iterrows():
                f.write(f"  Time Gap {gap} months:\n")
                f.write(f"    Samples: {row[('ssim', 'count')]}\n")
                f.write(
                    f"    SSIM: {row[('ssim', 'mean')]:.4f} ± {row[('ssim', 'std')]:.4f} (min: {row[('ssim', 'min')]:.4f}, max: {row[('ssim', 'max')]:.4f})\n")
                f.write(
                    f"    MSE: {row[('mse', 'mean')]:.4f} ± {row[('mse', 'std')]:.4f} (min: {row[('mse', 'min')]:.4f}, max: {row[('mse', 'max')]:.4f})\n\n")

            f.write("\nPrediction Gap Statistics:\n")
            for gap, row in analysis['prediction_gap_stats'].iterrows():
                f.write(f"  Prediction Gap {gap} months:\n")
                f.write(f"    Samples: {row[('ssim', 'count')]}\n")
                f.write(f"    SSIM: {row[('ssim', 'mean')]:.4f} ± {row[('ssim', 'std')]:.4f}\n")
                f.write(f"    MSE: {row[('mse', 'mean')]:.4f} ± {row[('mse', 'std')]:.4f}\n\n")

            f.write("\nRelative Position Statistics:\n")
            for pos, row in analysis['position_stats'].iterrows():
                f.write(f"  Position {pos:.1f}:\n")
                f.write(f"    Samples: {row[('ssim', 'count')]}\n")
                f.write(f"    SSIM: {row[('ssim', 'mean')]:.4f} ± {row[('ssim', 'std')]:.4f}\n")
                f.write(f"    MSE: {row[('mse', 'mean')]:.4f} ± {row[('mse', 'std')]:.4f}\n\n")


# Main function to run the evaluation
def evaluate_interpolation(model_path, csv_path, device="cuda:0"):
    """
    Main function to run interpolation evaluation

    Args:
        model_path: Path to the model checkpoint
        csv_path: Path to the CSV file with dataset information
        device: Device to run computations on

    Returns:
        DataFrame: Results of the evaluation
    """

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
    
    dm.setup('test')
    
    dataset = dm.test_ds

    # Load model
    model = load_model(model_path, device)
    model.gaussian_diffusion = GaussianDiffusion(model.timesteps_args, device=device)
    # Create evaluator and run evaluation
    evaluator = InterpolationEvaluator(model, dataset, device)
    results = evaluator.evaluate_all()

    # Save results
    evaluator.save_results(results)

    return results


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate interpolation accuracy for MRI scans")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cpu', 'cuda', 'mps')")
    parser.add_argument("--output_dir", type=str, default="interpolation_results", help="Directory to save results")

    args = parser.parse_args()

    device = torch.device(args.device)
    results = evaluate_interpolation(args.model_path, args.csv_path, device)

    print(f"Evaluation complete. Results saved to {args.output_dir}")