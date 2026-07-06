# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.11",
#     "matplotlib==3.11.0",
#     "numpy==2.5.0",
#     "pandas==3.0.3",
#     "pipeline==0.1.0",
#     "pydantic-ai-slim==2.5.0",
#     "ruptures==1.0.6",
#     "scikit-learn==1.9.0",
#     "scipy==1.18.0",
#     "torch==2.12.1",
#     "xarray==2026.4.0",
# ]
# [tool.marimo.venv]
# path = "/home/users/thopri/micromamba/envs/LatentTCN"      
# writable = false   
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Subset
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
    import os
    import glob
    import xarray as xr
    from sklearn.model_selection import KFold
    import copy
    import random


@app.class_definition
class OceanTippingDataset(Dataset):
    def __init__(self, csv_file):
        """
        Custom Dataset for reading sequential ocean ensemble data.
        """
        # 1. Load data from the CSV
        df = pd.read_csv(csv_file)

        # 2. Extract input features (X)
        # Drop columns that are markers or targets
        feature_cols = ['SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var','SSS_Rolling_Var','SSH_Grad_Rolling_Var']
        X_data = df[feature_cols].values # Converts to a numpy array

        # 3. Extract continuous target countdown (Y) and masking (U)
        Y_data = df['Time_To_Tip'].values
        U_data = df['Event_Mask'].values

        # 4. Convert everything to PyTorch Tensors
        # TCN models expect float32 for input features and targets
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_data, dtype=torch.float32)
        self.U = torch.tensor(U_data, dtype=torch.float32)

        # 5. Add Batch/Sequence Dimensions
        # Since this CSV represents 1 entire ensemble member sequence, 
        # we expand dimensions so it fits the expected TCN shape: 
        # X: (1, seq_len, num_features) | Y & U: (1, seq_len)
        self.X = self.X.unsqueeze(0)
        self.Y = self.Y.unsqueeze(0)
        self.U = self.U.unsqueeze(0)

    def __len__(self):
        # In this sequential setup, 1 CSV = 1 batch sample sequence
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.U[idx]


@app.function
# --- Pipeline Ingestion Hook ---
def create_ocean_dataloaders(csv_path, batch_size=1):
    """
    Instantiates the dataset and packages it inside a clean DataLoader framework.
    """
    dataset = OceanTippingDataset(csv_path)

    # CRITICAL: shuffle=False for sequence modeling. 
    # We want the time steps to remain in precise chronological order (1950 -> 2100).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


@app.class_definition
class ChainedCausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ChainedCausalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.p = dropout

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x, force_mc_dropout=False):
        # Layer 1
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = nn.functional.dropout(out, p=self.p, training=self.training or force_mc_dropout)

        # Layer 2
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = nn.functional.dropout(out, p=self.p, training=self.training or force_mc_dropout)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


@app.class_definition
class Chomp1d(nn.Module):
    """Slices off trailing padding to force causal temporal filtering."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


@app.class_definition
class TemporalWeibullRegressor(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalWeibullRegressor, self).__init__()
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            self.layers.append(ChainedCausalBlock(in_channels, out_channels, kernel_size, stride=1,
                                                 dilation=dilation_size, padding=padding, dropout=dropout))

        self.linear = nn.Linear(num_channels[-1], 2)

    def forward(self, x, force_mc_dropout=False):
        x_transposed = x.transpose(1, 2)

        features = x_transposed
        for layer in self.layers:
            features = layer(features, force_mc_dropout=force_mc_dropout)

        features = features.transpose(1, 2)
        raw_outputs = self.linear(features)
        raw_alpha = raw_outputs[..., 0]
        raw_beta = raw_outputs[..., 1]

        alpha = torch.clamp(torch.exp(raw_alpha), min=1e-3, max=1e5)
        beta = torch.clamp(nn.functional.softplus(raw_beta) + 1.0, min=1.001, max=50.0)

        return alpha, beta


@app.class_definition
# 1. Instantiate Custom Loss Framework
class WeibullNLLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(WeibullNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, alpha, beta, y, u):
        # Clip inputs slightly inside the loss function for safety
        alpha = torch.clamp(alpha, min=self.eps)
        beta = torch.clamp(beta, min=self.eps)
        y = torch.clamp(y, min=self.eps)

        log_y_div_alpha = torch.log(y) - torch.log(alpha)

        event_term = u * (torch.log(beta) - torch.log(alpha) + (beta - 1.0) * log_y_div_alpha)

        # Guard against large powers crashing into infinity
        ratio = torch.clamp(y / alpha, max=100.0) 
        survival_term = - torch.pow(ratio, beta)

        return torch.mean(-(event_term + survival_term))


@app.function
def train_weibull_tcn(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda'):
    """
    Standard training and validation loop for the TemporalWeibullRegressor.

    Args:
        model: The TemporalWeibullRegressor instance.
        train_loader: DataLoader containing (features, true_ttt, event_indicators) for training.
        val_loader: DataLoader containing validation ensemble members.
    """
    model = model.to(device)
    criterion = WeibullNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    # Adjust learning rate dynamically if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
# Track the metrics across epochs
    train_loss_history = []
    val_loss_history = []
    # Track the best validation loss and model state
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    print("Starting training pipeline...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y, batch_u in train_loader:
            # Move inputs and targets to execution device (GPU/CPU)
            batch_x = batch_x.to(device) # Shape: (batch, seq_len, num_features)
            batch_y = batch_y.to(device) # Shape: (batch, seq_len)
            batch_u = batch_u.to(device) # Shape: (batch, seq_len)

            optimizer.zero_grad()

            # Forward pass through the causal TCN
            alpha, beta = model(batch_x)

            # Compute Weibull Negative Log-Likelihood over the full time series
            loss = criterion(alpha, beta, batch_y, batch_u)

            # Backpropagation
            loss.backward()

            # Gradient clipping protects against exploding gradients caused by sharp spikes in beta
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss) # Save to history
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, batch_u in val_loader:
                batch_x, batch_y, batch_u = batch_x.to(device), batch_y.to(device), batch_u.to(device)

                alpha, beta = model(batch_x)
                loss = criterion(alpha, beta, batch_y, batch_u)
                running_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss) # Save to history
        # Step the learning rate scheduler based on validation performance
        scheduler.step(epoch_val_loss)

        # Check if this epoch produced the lowest validation loss seen so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Create an independent deep copy of the model weights
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1


        # Output epoch performance metrics
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train NLL: {epoch_train_loss:.4f} | Val NLL: {epoch_val_loss:.4f}")
        # Before returning, restore the model's weights to its best epoch performance
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"--> Restored best model weights from Epoch {best_epoch:02d} with Val NLL: {best_val_loss:.4f}")
    return train_loss_history, val_loss_history


@app.function
def predict_time_to_tip(alpha, beta):
    """
    Computes the analytical median and the 10th/90th percentiles of 
    the Weibull distribution, converting to standard arrays without 
    relying on PyTorch's internal NumPy C-bindings.
    """
    # 1. Ensure tensors are detached from any graph and on CPU
    alpha = alpha.detach().cpu()
    beta = beta.detach().cpu()

    # 2. Compute the percentiles
    median_ttt = alpha * torch.pow(torch.log(torch.tensor(2.0)), 1.0 / beta)
    p10_ttt = alpha * torch.pow(torch.log(torch.tensor(1.1111)), 1.0 / beta) 
    p90_ttt = alpha * torch.pow(torch.log(torch.tensor(10.0)), 1.0 / beta)    

    # 3. Safe alternative conversion: Convert directly to Python lists, 
    # then let the external numpy library package them into arrays in Cell 7.
    return np.array(median_ttt.tolist()), np.array(p10_ttt.tolist()), np.array(p90_ttt.tolist())


@app.function
def run_inference_and_plot(work_dir, models_list, test_loader, cv_train_histories, cv_val_histories, mc_iterations=50, num_plots=None, device='cpu', band_threshold=5.0):
    """
    Runs cross-model ensemble inference embedded with MC Dropout passes.
    Generates independent diagnostic dashboards for validation members.

    Parameters:
    -----------
    num_plots : int or None
        If None, plots all validation members.
        If an integer, randomly selects that many unique members to plot.
    band_threshold : float
        The width threshold for the 25th-75th percentile band to trigger a vertical line.
    """
    for model in models_list:
        model.eval()
        model.to(device)

    # 1. Collect full evaluation statistics per member trajectory across the loader
    member_data = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, _) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            true_seq = batch_y.detach().cpu().numpy()[0] # shape: (seq_len,)

            # --- DYNAMIC INDIVIDUAL MEMBER TIPPING DETECTION ---
            zero_indices = np.where(true_seq <= 0.05)[0]
            member_tipping_idx = zero_indices[0] if len(zero_indices) > 0 else np.argmin(true_seq)

            member_mc_medians = []
            for model in models_list:
                for _ in range(mc_iterations):
                    alpha, beta = model(batch_x, force_mc_dropout=True)
                    median, _, _ = predict_time_to_tip(alpha.cpu(), beta.cpu())
                    member_mc_medians.append(median[0])

            member_mc_medians = np.array(member_mc_medians) # shape: (Folds*MC, seq_len)

            # Compute trajectory distribution profiles
            m_median = np.median(member_mc_medians, axis=0)
            m_p10 = np.percentile(member_mc_medians, 10, axis=0)
            m_p90 = np.percentile(member_mc_medians, 90, axis=0)
            m_p25 = np.percentile(member_mc_medians, 25, axis=0)
            m_p75 = np.percentile(member_mc_medians, 75, axis=0)

            # Compute residual errors directly from the stochastic passes
            member_residuals = member_mc_medians - true_seq
            m_res_median = m_median - true_seq

            m_res_p10 = np.percentile(member_residuals, 10, axis=0)
            m_res_p90 = np.percentile(member_residuals, 90, axis=0)
            m_res_p25 = np.percentile(member_residuals, 25, axis=0)
            m_res_p75 = np.percentile(member_residuals, 75, axis=0)

            # Pack all records for this validation member
            member_data.append({
                'member_id': batch_idx + 1,
                'true_seq': true_seq,
                'tipping_idx': member_tipping_idx,
                'pred_median': m_median,
                'pred_p10': m_p10, 'pred_p90': m_p90,
                'pred_p25': m_p25, 'pred_p75': m_p75,
                'res_median': m_res_median,
                'res_p10': m_res_p10, 'res_p90': m_res_p90,
                'res_p25': m_res_p25, 'res_p75': m_res_p75,
                'flat_residuals': m_res_median # For individual histogram
            })

    # 2. Determine which validation records to plot
    total_members = len(member_data)
    indices_to_plot = list(range(total_members))

    if num_plots is not None and num_plots < total_members:
        indices_to_plot = random.sample(indices_to_plot, num_plots)
        print(f"--> Randomly selected {num_plots} out of {total_members} validation members for visualization.")

    time_steps = np.arange(len(member_data[0]['true_seq']))

    # 3. Dynamic Multi-Dashboard Canvas Loop
    for idx in indices_to_plot:
        m = member_data[idx]
        tip_x = time_steps[m['tipping_idx']]
        tip_y = m['true_seq'][m['tipping_idx']]

        with plt.style.context('default'):
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.patch.set_facecolor('white')
            fig.suptitle(f"Diagnostics: Validation Member #{m['member_id']}", fontsize=16, fontweight='bold', color='black')

            # --- Plot A: Aggregated Loss Convergence Profiles ---
            for f_idx, (t_hist, v_hist) in enumerate(zip(cv_train_histories, cv_val_histories)):
                axs[0, 0].plot(t_hist, color='royalblue', alpha=0.3, label='Train Folds' if f_idx == 0 else "")
                axs[0, 0].plot(v_hist, color='darkorange', alpha=0.3, linestyle='--', label='Val Folds' if f_idx == 0 else "")
            axs[0, 0].set_title('Cross-Validation Loss Convergence', color='black')
            axs[0, 0].set_xlabel('Epochs', color='black')
            axs[0, 0].set_ylabel('NLL Loss', color='black')
            axs[0, 0].grid(True, alpha=0.3)
            axs[0, 0].legend()

            # --- Plot B: Member Specific Horizon Tracking ---
            axs[0, 1].plot(time_steps, m['true_seq'], label='True TTT', color='green', lw=2, linestyle=':')
            axs[0, 1].plot(time_steps, m['pred_median'], label='Ensemble Median TTT', color='crimson', lw=2)

            # Layered Percentile Shading for Predictions
            axs[0, 1].fill_between(time_steps, m['pred_p10'], m['pred_p90'], color='crimson', alpha=0.15, label='10-90% Uncertainty')
            axs[0, 1].fill_between(time_steps, m['pred_p25'], m['pred_p75'], color='crimson', alpha=0.25, label='25-75% Interquartile')
            # --- UPDATED: Find where the band drops AND stays below the threshold ---
            iqr_band = m['pred_p75'] - m['pred_p25']
            
            # Compute a backward cumulative maximum of the band width
            # This represents the maximum band size from time step 't' to the end of the simulation
            suffix_max_band = np.maximum.accumulate(iqr_band[::-1])[::-1]
            
            # Find the first index where the future maximum is below the threshold
            below_threshold_indices = np.where(suffix_max_band < band_threshold)[0]
            
            if len(below_threshold_indices) > 0:
                narrow_idx = below_threshold_indices[0]
                narrow_x = time_steps[narrow_idx]
                axs[0, 1].axvline(x=narrow_x, color='teal', linestyle='-.', lw=2, 
                                  label=f'Band stays < {band_threshold} yrs (t={narrow_x})')
            axs[0, 1].axvline(x=tip_x, color='black', linestyle='--', alpha=0.7, label='Tipping Horizon')
            axs[0, 1].scatter(tip_x, tip_y, color='gold', edgecolor='black', s=200, marker='*', zorder=5, label='Tipping Event')

            axs[0, 1].set_title('Time-to-Tip Horizon Tracking', color='black')
            axs[0, 1].set_xlabel('Simulation Progress (Years)', color='black')
            axs[0, 1].set_ylabel('Years Remaining', color='black')
            axs[0, 1].grid(True, alpha=0.3)
            axs[0, 1].legend()

            # --- Plot C: Member Specific Residual Shaded Tracking ---
            axs[1, 0].plot(time_steps, m['res_median'], color='purple', lw=2, label='Ensemble Error')
            axs[1, 0].axhline(0, color='black', linestyle='--', alpha=0.7)

            # Layered Percentile Shading for Residual Errors
            axs[1, 0].fill_between(time_steps, m['res_p10'], m['res_p90'], color='purple', alpha=0.12, label='10-90% Error Spread')
            axs[1, 0].fill_between(time_steps, m['res_p25'], m['res_p75'], color='purple', alpha=0.22, label='25-75% Error Spread')

            axs[1, 0].axvline(x=tip_x, color='black', linestyle='--', alpha=0.4)

            axs[1, 0].set_title('Prediction Error Over Timeline', color='black')
            axs[1, 0].set_xlabel('Simulation Progress (Years)', color='black')
            axs[1, 0].set_ylabel('Error (Years)', color='black')
            axs[1, 0].grid(True, alpha=0.3)
            axs[1, 0].legend()

            # --- Plot D: Member Specific Error Frequency Distribution Histogram ---
            axs[1, 1].hist(m['flat_residuals'], bins=15, color='seagreen', edgecolor='black', alpha=0.7)
            axs[1, 1].axvline(0, color='black', linestyle='--', alpha=0.7)
            axs[1, 1].set_title('Distribution of Model Errors', color='black')
            axs[1, 1].set_xlabel('Error (Years)', color='black')
            axs[1, 1].set_ylabel('Frequency Count', color='black')
            axs[1, 1].grid(True, alpha=0.3)

            for ax in axs.flat:
                ax.set_facecolor('white')
                ax.tick_params(colors='black')

            plt.tight_layout()
            plt.savefig(f"{work_dir}/ML_diagnostics_val_member_{m['member_id']}.png")
            plt.show()


@app.class_definition
class MultiMemberOceanDataset(Dataset):
    def __init__(self, search_pattern,work_dir):

        all_x, all_y, all_u = [], [], []

            # 1. Find all matching files (e.g., ensemble_01.csv, ensemble_02.csv, etc.)
        search_pattern = os.path.join(work_dir, search_pattern)
        file_paths = glob.glob(search_pattern)
        # 2. Defensive check to prevent empty torch.cat crashes
        if len(file_paths) == 0:
            raise FileNotFoundError(
                f"No files found matching pattern '{search_pattern}' in directory '{os.getcwd()}'"
            )

        for path in file_paths:
            df = pd.read_csv(path)

            # Extract features and targets
            x = torch.tensor(df[['SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var','SSS_Rolling_Var','SSH_Grad_Rolling_Var']].values, dtype=torch.float32)
            y = torch.tensor(df['Time_To_Tip'].values, dtype=torch.float32)
            u = torch.tensor(df['Event_Mask'].values, dtype=torch.float32)

            all_x.append(x.unsqueeze(0)) # Shape (1, 151, 4)
            all_y.append(y.unsqueeze(0)) # Shape (1, 151)
            all_u.append(u.unsqueeze(0)) # Shape (1, 151)

        # Cat combines them along the Batch axis (dim 0)
        self.X = torch.cat(all_x, dim=0) # Shape: (Num_Members, 151, 4)
        self.Y = torch.cat(all_y, dim=0) # Shape: (Num_Members, 151)
        self.U = torch.cat(all_u, dim=0) # Shape: (Num_Members, 151)

    def __len__(self):
        return self.X.shape[0] # Returns total number of ensemble members

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.U[idx]


@app.class_definition
class SpatialOceanDataset(Dataset):
    """
    2D counterpart to MultiMemberOceanDataset.

    Instead of one (1, seq_len, num_features) sequence per ensemble member
    (as produced by extract_single_point_features_for_all_ensembles), this
    loads the gridded (time, y, x) NetCDF files produced by
    process_spg_features_and_targets and unpacks *every usable ocean pixel*
    of *every ensemble member* into its own (seq_len, num_features) sequence.

    The TCN/Weibull model itself needs no changes to consume this: it only
    ever sees a batch of independent (seq_len, num_features) sequences, and
    doesn't care whether that batch axis indexes ensemble members (1D case)
    or (ensemble member, y, x) pixels (2D case).

    Two bookkeeping arrays are kept alongside X/Y/U so that:
      - K-Fold splitting can be done by *ensemble member* (see main block)
        rather than by pixel, to avoid leaking neighbouring pixels from the
        same member across the train/val boundary.
      - Predictions can be scattered back onto the (y, x) grid for mapping.
    """
    feature_vars = ['SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var','SSS_Rolling_Var','SSH_Grad_Rolling_Var']

    def __init__(self, work_dir, search_pattern="ensemble_spg_spatial_*.nc", member_ids=None):
        search_path = os.path.join(work_dir, search_pattern)
        file_paths = sorted(glob.glob(search_path))
        if len(file_paths) == 0:
            raise FileNotFoundError(
                f"No files found matching pattern '{search_path}'"
            )

        all_x, all_y, all_u = [], [], []
        pixel_member, pixel_yx = [], []

        for path in file_paths:
            # Ensemble ID is inferred from the filename, e.g.
            # 'ensemble_spg_spatial_07.nc' -> 7
            digits = ''.join(ch for ch in os.path.basename(path) if ch.isdigit())
            member_id = int(digits) if digits else -1

            ds = xr.open_dataset(path)
            feats = np.stack([ds[v].values for v in self.feature_vars], axis=-1)  # (T, Y, X, F)
            ttt = ds['Time_To_Tip'].values   # (T, Y, X)
            mask = ds['Event_Mask'].values   # (T, Y, X)
            ds.close()

            # Match the bfill/ffill convention used in the 1D point-extraction
            # pipeline: this patches rolling-window edge NaNs (e.g. the first
            # and last couple of years of SST_Rolling_Var) without touching
            # genuine land NaNs, which remain NaN across the *entire* record.
            for f_idx in range(feats.shape[-1]):
                da = xr.DataArray(feats[..., f_idx], dims=('time', 'y', 'x'))
                feats[..., f_idx] = da.bfill(dim='time').ffill(dim='time').values

            # Ocean pixels only: land shows up as NaN across every timestep.
            # Anything still NaN after the fill above is land (or a pixel with
            # no valid data at all), so we drop it here rather than feeding
            # NaNs into the TCN.
            usable = ~np.isnan(feats).any(axis=(0, 3)) & ~np.isnan(ttt).any(axis=0)
            ys, xs = np.where(usable)

            if len(ys) == 0:
                print(f"  Member {member_id:02d}: no usable ocean pixels found, skipping file.")
                continue

            # Vectorized gather + reorder to (Pixels, Time, Features)
            pixel_feats = feats[:, ys, xs, :].transpose(1, 0, 2)
            pixel_ttt = ttt[:, ys, xs].transpose(1, 0)
            pixel_mask = mask[:, ys, xs].transpose(1, 0)

            all_x.append(torch.tensor(pixel_feats, dtype=torch.float32))
            all_y.append(torch.tensor(pixel_ttt, dtype=torch.float32))
            all_u.append(torch.tensor(pixel_mask, dtype=torch.float32))
            pixel_member.extend([member_id] * len(ys))
            pixel_yx.extend(list(zip(ys.tolist(), xs.tolist())))

            print(f"  Member {member_id:02d}: {len(ys)} usable ocean pixels loaded.")

        if len(all_x) == 0:
            raise RuntimeError("No usable ocean pixels found across any input file.")

        self.X = torch.cat(all_x, dim=0)  # (N_pixels_total, seq_len, num_features)
        self.Y = torch.cat(all_y, dim=0)
        self.U = torch.cat(all_u, dim=0)
        self.pixel_member = np.array(pixel_member)
        self.pixel_yx = np.array(pixel_yx)  # (N_pixels_total, 2) -> (y, x)

        if member_ids is not None:
            keep = np.isin(self.pixel_member, member_ids)
            self.X, self.Y, self.U = self.X[keep], self.Y[keep], self.U[keep]
            self.pixel_member = self.pixel_member[keep]
            self.pixel_yx = self.pixel_yx[keep]

        print(f"SpatialOceanDataset: {self.X.shape[0]} total pixel-sequences "
              f"from {len(np.unique(self.pixel_member))} ensemble members.")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.U[idx]


@app.function
def process_spg_features_and_targets(
    base_dir, work_dir, scale_factor=0.2, min_absolute_floor=100.0
):
    """Processes all 40 ensembles for both 2D features and 2D spatial tipping targets,

    saving one consolidated NetCDF file per ensemble member.
    """
    # --- Define Subpolar Gyre Bounding Box (Adjusted for ORCA/NEMO native coordinate grids) ---
    LAT_MIN, LAT_MAX = 60.819, 60.819  # 45.0, 70.0
    LON_MIN, LON_MAX = -56.506, -56.506  # -60.0, 0.0
    BASELINE_ST_YEAR = 1950
    BASELINE_END_YEAR = 1980

    ensemble_members = range(1, 41)
    var_mappings = {
        "temp": "*votemper.nc",
        "salinity": "*vosaline.nc",
        "ssh": "*zossq.nc",
        "mld": "*somxl010.nc",
    }

    # Set up scenario timelines
    timeline = [("HIST2", y) for y in range(1950, 2015)] + [
        ("SSP370", y) for y in range(2015, 2101)
    ]

    for ens in ensemble_members:
        print(f"\n=========================================")
        print(f"Processing Ensemble Member: {ens:02d}")
        print(f"=========================================")

        annual_features = []
        annual_mlds = []

        for experiment, year in timeline:
            year_dir = os.path.join(
                base_dir, experiment, str(ens), "OCN", "yearly", str(year)
            )
            files = {}
            for var_name, pattern in var_mappings.items():
                found_files = sorted(glob.glob(os.path.join(year_dir, pattern)))
                if found_files:
                    files[var_name] = found_files[0]

            if len(files) < 4:
                print(
                    f"  Missing files for year {year} ({experiment}). Skipping..."
                )
                continue

            # Open Datasets
            ds_temp = xr.open_dataset(files["temp"])
            ds_salt = xr.open_dataset(files["salinity"])
            ds_ssh = xr.open_dataset(files["ssh"])
            ds_mld = xr.open_dataset(files["mld"])

            # Determine Depth Dimension Name Dynamically
            depth_dim = (
                "depth"
                if "depth" in ds_temp.dims
                else ("olevel" if "olevel" in ds_temp.dims else "deptht")
            )

            # 1. Spatial Masking & Cropping Setup
            if LAT_MIN == LAT_MAX and LON_MIN == LON_MAX:
                print(
                    f"   Target coordinates identify a single point: ({LAT_MIN}, {LON_MIN}). Extracting local neighborhood..."
                )

                # Compute absolute Euclidean distance from every 2D grid cell to the single point target
                distance = np.sqrt(
                    (ds_temp["nav_lat"] - LAT_MIN) ** 2
                    + (ds_temp["nav_lon"] - LON_MIN) ** 2
                )

                # Find the central (y, x) indices of the single minimum distance cell
                min_idx = distance.argmin()
                y_center, x_center = np.unravel_index(
                    min_idx.values, distance.shape
                )

                # Define a 3x3 local bounding box matrix slice around the point to preserve gradient contexts
                y_slice = slice(
                    max(0, y_center - 1), min(ds_temp.dims["y"], y_center + 2)
                )
                x_slice = slice(
                    max(0, x_center - 1), min(ds_temp.dims["x"], x_center + 2)
                )

                # Crop raw inputs directly using structural integer location index slicing (.isel)
                ds_t_crop = ds_temp.isel(y=y_slice, x=x_slice)
                ds_s_crop = ds_salt.isel(y=y_slice, x=x_slice)
                ds_ssh_crop = ds_ssh.isel(y=y_slice, x=x_slice)
                ds_mld_crop = ds_mld.isel(y=y_slice, x=x_slice)

                is_single_point_run = True
            else:
                spatial_mask = (
                    (ds_temp["nav_lat"] >= LAT_MIN)
                    & (ds_temp["nav_lat"] <= LAT_MAX)
                    & (ds_temp["nav_lon"] >= LON_MIN)
                    & (ds_temp["nav_lon"] <= LON_MAX)
                )

                ds_t_crop = ds_temp.where(spatial_mask, drop=True)
                ds_s_crop = ds_salt.where(spatial_mask, drop=True)
                ds_ssh_crop = ds_ssh.where(spatial_mask, drop=True)
                ds_mld_crop = ds_mld.where(spatial_mask, drop=True)

                is_single_point_run = False

            # Extract surface metrics
            surf_temp = ds_t_crop["votemper"].isel({depth_dim: 0}).rename("SST")
            surf_salt = ds_s_crop["vosaline"].isel({depth_dim: 0}).rename("SSS")
            surf_ssh = ds_ssh_crop["zossq"]
            mld_val = ds_mld_crop["somxl010"]

            # 2. Compute Spatial Gradient on the *Cropped* Field
            ssh_values = surf_ssh.values
            ssh_grad_y, ssh_grad_x = np.gradient(ssh_values, axis=(1, 2))
            ssh_grad_mag = np.sqrt(ssh_grad_y**2 + ssh_grad_x**2)

            surf_grad = xr.DataArray(
                ssh_grad_mag,
                coords=surf_ssh.coords,
                dims=surf_ssh.dims,
                name="SSH_Gradient",
            )

            # Merge features for this specific year
            ds_year_spatial = xr.merge(
                [surf_temp, surf_salt, surf_grad], compat="override"
            ).compute()

            # --- DOWN-SLICE TO EXACTLY 1 PIXEL IF SINGLE POINT RUN ---
            if is_single_point_run:
                y_mid = 1 if ds_year_spatial.dims["y"] >= 3 else 0
                x_mid = 1 if ds_year_spatial.dims["x"] >= 3 else 0

                # Slicing as lists [y_mid] and [x_mid] keeps the dimensions with a length of 1
                ds_year_spatial = ds_year_spatial.isel(y=[y_mid], x=[x_mid])
                mld_val = mld_val.isel(y=[y_mid], x=[x_mid])

            annual_features.append(ds_year_spatial)
            annual_mlds.append(mld_val.compute())

        print("  Concatenating time series along temporal dimension...")
        spg_features = xr.concat(annual_features, dim="time_counter").rename(
            {"time_counter": "time"}
        )
        spg_mld = xr.concat(annual_mlds, dim="time_counter").rename(
            {"time_counter": "time"}
        )

        # 3. Calculate Anomalies and Z-Scores across the grid axis
        baseline = spg_features.sel(
            time=slice(str(BASELINE_ST_YEAR), str(BASELINE_END_YEAR))
        )

        sst_baseline_mean = baseline["SST"].mean("time")
        sst_baseline_std = baseline["SST"].std("time")
        sst_baseline_std = xr.where(sst_baseline_std == 0, 1.0, sst_baseline_std)

        sss_baseline_mean = baseline["SSS"].mean("time")
        sss_baseline_std = baseline["SSS"].std("time")
        sss_baseline_std = xr.where(sss_baseline_std == 0, 1.0, sss_baseline_std)

        ssh_grad_baseline_mean = baseline["SSH_Gradient"].mean("time")
        ssh_grad_baseline_std = baseline["SSH_Gradient"].std("time")
        ssh_grad_baseline_std = xr.where(
            ssh_grad_baseline_std == 0, 1.0, ssh_grad_baseline_std
        )

        sst_zscore = (spg_features["SST"] - sst_baseline_mean) / sst_baseline_std
        sss_zscore = (spg_features["SSS"] - sss_baseline_mean) / sss_baseline_std
        ssh_grad_zscore = (
            spg_features["SSH_Gradient"] - ssh_grad_baseline_mean
        ) / ssh_grad_baseline_std

        # 60-month (5-year) rolling standard deviations for Early Warning Signals (EWS)
        sst_rolling_var = sst_zscore.rolling(
            time=60, center=True, min_periods=12
        ).std()
        sss_rolling_var = sss_zscore.rolling(
            time=60, center=True, min_periods=12
        ).std()
        ssh_grad_rolling_var = ssh_grad_zscore.rolling(
            time=60, center=True, min_periods=12
        ).std()

        # Combine into an intermediate annual dataset
        features_ds = xr.Dataset(
            {
                "SST_Anom": sst_zscore,
                "SSS_Anom": sss_zscore,
                "SSH_Gradient": ssh_grad_zscore,
                "SST_Rolling_Var": sst_rolling_var,
                "SSS_Rolling_Var": sss_rolling_var,
                "SSH_Grad_Rolling_Var": ssh_grad_rolling_var,
            }
        ).resample(
            time="1YS"
        ).mean()  # Downsample to Annual Mean

        # 4. Generate Spatial Tipping Targets Vectorized Block
        print(
            "  Computing pixel-by-pixel spatial tipping targets using a localized climatology..."
        )
        annual_max_mld = spg_mld.resample(time="1YS").max()

        time_steps = annual_max_mld["time"].values
        years = np.array([t.year for t in time_steps])

        mld_matrix = annual_max_mld.values  # Always shape (Time, Y, X) because of length-1 list slicing
        time_len, y_len, x_len = mld_matrix.shape

        # --- DYNAMIC CLIMATOLOGY THRESHOLD GENERATION ---
        baseline_years_count = min(
            BASELINE_END_YEAR - BASELINE_ST_YEAR + 1, time_len
        )
        historical_baseline_mld = mld_matrix[:baseline_years_count, :, :]
        mld_climatology_map = np.nanmean(historical_baseline_mld, axis=0)

        dynamic_thresholds = mld_climatology_map * scale_factor
        dynamic_thresholds = np.where(
            dynamic_thresholds < min_absolute_floor,
            min_absolute_floor,
            dynamic_thresholds,
        )

        ttt_matrix = np.zeros_like(mld_matrix)
        mask_matrix = np.ones_like(mld_matrix)

        # Loop through space coordinates to evaluate tipping points using spatial thresholds
        for y in range(y_len):
            for x in range(x_len):
                pixel_mld = mld_matrix[:, y, x]
                pixel_threshold = dynamic_thresholds[y, x]

                if np.isnan(pixel_mld).all() or np.isnan(pixel_threshold):
                    ttt_matrix[:, y, x] = np.nan
                    mask_matrix[:, y, x] = 0
                    continue

                tipping_idx = None
                for t in range(time_len):
                    if pixel_mld[t] < pixel_threshold:
                        if np.all(pixel_mld[t:] < pixel_threshold):
                            tipping_idx = t
                            break

                if tipping_idx is not None:
                    for t in range(time_len):
                        if t <= tipping_idx:
                            ttt_matrix[t, y, x] = years[tipping_idx] - years[t]
                            mask_matrix[t, y, x] = 1
                        else:
                            ttt_matrix[t, y, x] = 0
                            mask_matrix[t, y, x] = 1
                else:
                    for t in range(time_len):
                        ttt_matrix[t, y, x] = years[-1] - years[t]
                        mask_matrix[t, y, x] = 0

        # Preserve structural maps tracking coordinates
        features_ds["nav_lat"] = annual_max_mld["nav_lat"]
        features_ds["nav_lon"] = annual_max_mld["nav_lon"]

        # Directly pack the standard 3D layout (No special squeezing or conditional handling required)
        features_ds["Time_To_Tip"] = (("time", "y", "x"), ttt_matrix)
        features_ds["Event_Mask"] = (("time", "y", "x"), mask_matrix)

        # Save consolidated spatial matrix file
        out_path = f"{work_dir}/ensemble_spg_spatial_{ens:02d}.nc"
        features_ds.to_netcdf(out_path)
        print(f"  Successfully saved spatial dataset to: {out_path}")


@app.function
def generate_ensemble_tipping_maps(models_list, target_ensemble_nc, work_dir, device='cpu',
                                    eval_years=[2026, 2036, 2046, 2056], mc_iterations=30):
    """
    2D replacement for generate_decadal_tipping_maps.

    Loads one gridded (time, y, x) evaluation file (as produced by
    process_spg_features_and_targets), flattens every ocean pixel into a
    parallel batch of sequences, and pushes that batch through *all* K-Fold
    models with MC-Dropout enabled -- the same ensembling strategy
    run_inference_and_plot uses for single-point diagnostics -- to produce
    both a median Time-To-Tip map and an uncertainty (IQR) map at each
    requested decadal snapshot.

    Parameters
    ----------
    models_list : list of trained TemporalWeibullRegressor instances (one per fold)
    target_ensemble_nc : path to a single ensemble_spg_spatial_NN.nc file
    device : 'cpu' or 'cuda' -- no more hardcoded .cuda()
    eval_years : snapshot years to plot
    mc_iterations : number of stochastic forward passes per fold model
    """
    for model in models_list:
        model.eval()
        model.to(device)

    ds = xr.open_dataset(target_ensemble_nc)

    # 1. Map configuration
    years = np.array([t.year for t in ds['time'].values])
    lat = ds['nav_lat'].values
    lon = ds['nav_lon'].values

    feature_list = ['SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var','SSS_Rolling_Var','SSH_Grad_Rolling_Var']
    data_matrix = np.stack([ds[f].values for f in feature_list], axis=-1)  # (T, Y, X, F)
    time_len, y_len, x_len, num_features = data_matrix.shape

    # 2. Same NaN handling convention as SpatialOceanDataset/training: bfill/ffill
    # patches rolling-window edge NaNs, leaving genuine land NaNs in place.
    for f_idx in range(num_features):
        da = xr.DataArray(data_matrix[..., f_idx], dims=('time', 'y', 'x'))
        data_matrix[..., f_idx] = da.bfill(dim='time').ffill(dim='time').values

    land_mask = np.isnan(data_matrix).any(axis=(0, 3))  # (Y, X) -- True over land/unusable pixels

    # Land pixels still need *some* finite value to pass through the TCN; fill with
    # zero (they get masked back out to NaN before plotting, so the value is inert).
    safe_matrix = np.nan_to_num(data_matrix, nan=0.0)

    # Flatten spatial field to treat pixels as parallel batch sequences: (Y*X, Time, Features)
    flattened_features = safe_matrix.transpose(1, 2, 0, 3).reshape(y_len * x_len, time_len, num_features)
    tensor_input = torch.tensor(flattened_features, dtype=torch.float32).to(device)

    # 3. MC-Dropout ensemble forward pass across every fold model
    all_medians = []
    with torch.no_grad():
        for model in models_list:
            for _ in range(mc_iterations):
                alpha, beta = model(tensor_input, force_mc_dropout=True)
                median_flat, _, _ = predict_time_to_tip(alpha, beta)  # (Y*X, Time)
                all_medians.append(median_flat)

    all_medians = np.stack(all_medians, axis=0)  # (Folds*MC, Y*X, Time)
    median_ttt_flat = np.median(all_medians, axis=0)                    # (Y*X, Time)
    iqr_ttt_flat = np.percentile(all_medians, 75, axis=0) - np.percentile(all_medians, 25, axis=0)

    # Reshape back to the geographical grid: (Time, Y, X)
    median_ttt_spatial = median_ttt_flat.reshape(y_len, x_len, time_len).transpose(2, 0, 1)
    iqr_ttt_spatial = iqr_ttt_flat.reshape(y_len, x_len, time_len).transpose(2, 0, 1)

    with plt.style.context('default'):

        # 4. Plot Snapshots Across Time -- median TTT (top row) and IQR uncertainty (bottom row)
        fig, axs = plt.subplots(2, len(eval_years), figsize=(20, 9), sharey=True)
        fig.patch.set_facecolor('white')
        for i, target_year in enumerate(eval_years):
            time_idx = np.where(years == target_year)[0][0]
    
            map_data = median_ttt_spatial[time_idx, :, :].copy()
            map_data[land_mask] = np.nan
            im_top = axs[0, i].pcolormesh(lon, lat, map_data, cmap='bwr_r', vmin=0, vmax=100)
            axs[0, i].set_title(f"Median predicted TTT: {target_year}")
            if i == 0: axs[0, i].set_ylabel("Latitude")
    
            unc_data = iqr_ttt_spatial[time_idx, :, :].copy()
            unc_data[land_mask] = np.nan
            im_bot = axs[1, i].pcolormesh(lon, lat, unc_data, cmap='viridis', vmin=0)
            axs[1, i].set_title(f"IQR uncertainty: {target_year}")
            axs[1, i].set_xlabel("Longitude")
            if i == 0: axs[1, i].set_ylabel("Latitude")
    
        # Layout presentation details
        cbar_top = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        fig.colorbar(im_top, cax=cbar_top, label="Years Remaining to Tipping Event")
        cbar_bot = fig.add_axes([0.92, 0.1, 0.015, 0.35])
        fig.colorbar(im_bot, cax=cbar_bot, label="25-75% IQR (Years)")
    
        out_path = f"{work_dir}/SPG_Decadal_Inference_Snapshots.png"
        plt.savefig(out_path, bbox_inches='tight')
        plt.show()

    return median_ttt_spatial, iqr_ttt_spatial, years, lat, lon


@app.cell
def _():
    if __name__ == "__main__":
        extract_from_CANARI = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"running on device: {device}")
        base_dir = "/gws/ssde/j25b/canari/shared/large-ensemble/priority"
        work_dir = "/gws/ssde/j25a/nemo/vol4/thopri/StateBased/CANARI_1D"

        if extract_from_CANARI:
            # This now always builds the gridded (time, y, x) NetCDF files, one
            # per ensemble member, rather than single-pixel CSVs.
            process_spg_features_and_targets(base_dir=base_dir, work_dir=work_dir)

        # 1. Load every usable ocean pixel from every ensemble member as an
        # independent training sequence. Bookkeeping (pixel_member, pixel_yx)
        # travels alongside X/Y/U for member-aware K-Fold splitting and for
        # scattering predictions back onto the map later.
        data = SpatialOceanDataset(work_dir=work_dir, search_pattern="ensemble_spg_spatial_*.nc")

        full_dataset = TensorDataset(data.X, data.Y, data.U)
        num_features = 6

        # 2. Setup K-Fold parameters -- split by ENSEMBLE MEMBER, not by pixel.
        # Splitting on flattened pixel indices would put neighbouring pixels
        # from the same member on both sides of the train/val boundary, which
        # leaks spatially-correlated information and inflates validation skill.
        # Holding out whole members instead tests genuine generalization to
        # unseen simulations, mirroring what the 1D per-member K-Fold did.
        n_splits = 5
        unique_members = np.unique(data.pixel_member)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=64)

        trained_models = []
        cv_train_histories = []
        cv_val_histories = []

        # 3. K-Fold Training Loop
        for fold, (train_member_pos, val_member_pos) in enumerate(kf.split(unique_members)):
            print(f"\n--- Training Fold {fold + 1} / {n_splits} ---")

            train_members = unique_members[train_member_pos]
            val_members = unique_members[val_member_pos]

            train_idx = np.where(np.isin(data.pixel_member, train_members))[0]
            val_idx = np.where(np.isin(data.pixel_member, val_members))[0]

            # Sub-allocations using PyTorch Subsets
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            # Batch size is small due to small number of ensembles (40)
            train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)

            # Initialize isolated network configurations per fold
            fold_model = TemporalWeibullRegressor(num_inputs=num_features, num_channels=[32, 64, 128])

            train_hist, val_hist = train_weibull_tcn(
                fold_model, train_loader, val_loader, epochs=30, lr=5e-5, device=device
            )

            trained_models.append(fold_model)
            cv_train_histories.append(train_hist)
            cv_val_histories.append(val_hist)

        # 4. Final Holdout Inference Evaluation Loader Setup
        # Using the final fold's held-out members. Plotting every held-out pixel
        # isn't useful, so a small random sample of individual pixel sequences
        # is drawn for the per-sequence diagnostic dashboards; the full spatial
        # picture is produced separately in step 6 below.
        rng = np.random.default_rng(64)
        val_sample_idx = rng.choice(val_idx, size=min(200, len(val_idx)), replace=False)
        val_subset_final = Subset(full_dataset, val_sample_idx)
        final_test_loader = DataLoader(val_subset_final, batch_size=1, shuffle=False)

        # 5. Generate metrics and distribution diagnostic curves for a handful
        # of individual validation pixels
        run_inference_and_plot(
            work_dir=work_dir,
            models_list=trained_models,
            test_loader=final_test_loader,
            cv_train_histories=cv_train_histories,
            cv_val_histories=cv_val_histories,
            mc_iterations=30, # Runs 30 iterations * 5 folds = 150 predictions per sequence step
            device=device,
            # num_plots=2,
        )

        # 6. Generate full spatial Time-To-Tip maps (median + uncertainty) at
        # decadal snapshots for one held-out ensemble member's gridded file.
        target_member = int(val_members[0])
        target_nc = f"{work_dir}/ensemble_spg_spatial_{target_member:02d}.nc"
        generate_ensemble_tipping_maps(
            models_list=trained_models,
            target_ensemble_nc=target_nc,
            work_dir=work_dir,
            device=device,
            eval_years=[2026, 2036, 2046, 2056],
            mc_iterations=30,
        )
    return


if __name__ == "__main__":
    app.run()
