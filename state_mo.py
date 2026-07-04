# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.11",
#     "matplotlib==3.11.0",
#     "numpy==2.5.0",
#     "pandas==3.0.3",
#     "pipeline==0.1.0",
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
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
    import os
    import glob
    import xarray as xr

    return (
        DataLoader,
        Dataset,
        TensorDataset,
        glob,
        nn,
        np,
        optim,
        os,
        pd,
        plt,
        random_split,
        torch,
        weight_norm,
        xr,
    )


@app.cell
def _(DataLoader, Dataset, pd, torch):
    class OceanTippingDataset(Dataset):
        def __init__(self, csv_file):
            """
            Custom Dataset for reading sequential ocean ensemble data.
            """
            # 1. Load data from the CSV
            df = pd.read_csv(csv_file)

            # 2. Extract input features (X)
            # Drop columns that are markers or targets
            feature_cols = ['SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var']
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


    return


@app.cell
def _(nn, weight_norm):
    class ChainedCausalBlock(nn.Module):
        """
        A 1D Causal Dilated Convolutional Block with Residual Connections.
        Ensures that the model cannot 'look ahead' into the future.
        """
        def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
            super(ChainedCausalBlock, self).__init__()
            # Dilated 1D Convolution
            self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            # Chomp layer removes the padding added to the end of the sequence to maintain causality
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)

            # Match dimensions for residual connection if channels differ
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)


    class Chomp1d(nn.Module):
        """Slices off trailing padding to force causal temporal filtering."""
        def __init__(self, chomp_size):
            super(Chomp1d, self).__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

    return (ChainedCausalBlock,)


@app.cell
def _(ChainedCausalBlock, nn, torch):
    class TemporalWeibullRegressor(nn.Module):
        def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
            """
            Args:
                num_inputs (int): Number of surface features (e.g., SST, SSS, SSH anomalies).
                num_channels (list): List of channel sizes per TCN layer (e.g., [32, 64, 128]).
                kernel_size (int): Temporal filter size.
            """
            super(TemporalWeibullRegressor, self).__init__()
            layers = []
            num_levels = len(num_channels)

            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = num_inputs if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]

                # Padding is calculated explicitly to preserve causal step dimensions
                padding = (kernel_size - 1) * dilation_size

                layers += [ChainedCausalBlock(in_channels, out_channels, kernel_size, stride=1,
                                              dilation=dilation_size, padding=padding, dropout=dropout)]

            self.tcn = nn.Sequential(*layers)

            # Final fully connected layer outputs 2 parameters per time step: alpha and beta
            self.linear = nn.Linear(num_channels[-1], 2)

        def forward(self, x):
            """
            Args:
                x (Tensor): Input tensor of shape (batch_size, seq_len, num_inputs)
            Returns:
                alpha, beta (Tensors): Parameters shaping the time-to-tip distribution at each step.
            """
            # PyTorch Conv1d expects (batch_size, channels, seq_len)
            x_transposed = x.transpose(1, 2)

            # Extract features across historical timeline
            features = self.tcn(x_transposed)

            # Transpose back to (batch_size, seq_len, hidden_dim) for regression
            features = features.transpose(1, 2)

            # Predict parameter representations
            raw_outputs = self.linear(features)
            raw_alpha = raw_outputs[..., 0]
            raw_beta = raw_outputs[..., 1]

            # Activations ensuring valid domain space constraints (> 0)
            alpha = torch.exp(raw_alpha)
            beta = nn.functional.softplus(raw_beta) + 1.0  # +1 encourages increasing failure rate

            return alpha, beta

    return (TemporalWeibullRegressor,)


@app.cell
def _(nn, torch):
    # 1. Instantiate Custom Loss Framework
    class WeibullNLLLoss(nn.Module):
        def __init__(self, eps=1e-6):
            super(WeibullNLLLoss, self).__init__()
            self.eps = eps

        def forward(self, alpha, beta, y, u):
            y = y + self.eps
            log_y_div_alpha = torch.log(y) - torch.log(alpha)
            event_term = u * (torch.log(beta) - torch.log(alpha) + (beta - 1.0) * log_y_div_alpha)
            survival_term = - torch.pow(y / alpha, beta)
            return torch.mean(-(event_term + survival_term))

    # # 2. Predictive Inference Function
    # def predict_time_to_tip(alpha, beta):
    #     """
    #     Computes the analytical median of the Weibull distribution.
    #     The median is robust against the long tail distributions common far away from the tipping year.
    #     """
    #     with torch.no_grad():
    #         median_ttt = alpha * torch.pow(torch.log(torch.tensor(2.0)), 1.0 / beta)
    #     return median_ttt
    return (WeibullNLLLoss,)


@app.cell
def _(WeibullNLLLoss, optim, torch):
    def train_weibull_tcn(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cuda'):
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

            # Output epoch performance metrics
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train NLL: {epoch_train_loss:.4f} | Val NLL: {epoch_val_loss:.4f}")
        return train_loss_history, val_loss_history

    return (train_weibull_tcn,)


@app.cell
def _(np, torch):
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


    return (predict_time_to_tip,)


@app.cell
def _(np, plt, predict_time_to_tip, torch):
    def run_inference_and_plot(model, test_loader, train_loss_history, val_loss_history, device='cpu'):
        """
        Runs inference on a holdout ensemble member/satellite run, extracts 
        residuals, and plots model diagnostics.
        """
        model.eval()
        model.to(device)

        all_true = []
        all_pred_med = []
        all_p10 = []
        all_p90 = []

        # 1. Run Inference
        with torch.no_grad():
            for batch_x, batch_y, _ in test_loader:
                batch_x = batch_x.to(device)

                # Predict Weibull parameters
                alpha, beta = model(batch_x)

                # Convert distribution to explicit year predictions (already clean now!)
                median, p10, p90 = predict_time_to_tip(alpha.cpu(), beta.cpu())

                # FIX: Convert the ground truth PyTorch tensor to a standard Python list first
                all_true.append(np.array(batch_y.detach().cpu().tolist()))
                all_pred_med.append(median)
                all_p10.append(p10)
                all_p90.append(p90)

        # Concatenate using standard numpy array manipulation, extracting the single sequence
        true_years = np.concatenate(all_true, axis=0)[0]       # Shape: (seq_len,)
        pred_median = np.concatenate(all_pred_med, axis=0)[0]   # Shape: (seq_len,)
        pred_p10 = np.concatenate(all_p10, axis=0)[0]          # Shape: (seq_len,)
        pred_p90 = np.concatenate(all_p90, axis=0)[0]          # Shape: (seq_len,)

        time_steps = np.arange(len(true_years))
        residuals = pred_median - true_years

        # 2. Plotting Dashboard
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Subpolar Gyre Time-to-Tip ML Diagnostics', fontsize=16, fontweight='bold')

        # --- Plot A: Loss Curves ---
        axs[0, 0].plot(train_loss_history, label='Train NLL Loss', color='royalblue', lw=2)
        axs[0, 0].plot(val_loss_history, label='Val NLL Loss', color='darkorange', lw=2, linestyle='--')
        axs[0, 0].set_title('Training & Validation Convergence')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Negative Log-Likelihood')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # --- Plot B: Continuous Forecast Horizon ---
        axs[0, 1].plot(time_steps, true_years, label='True Time-to-Tip', color='black', lw=2, linestyle=':')
        axs[0, 1].plot(time_steps, pred_median, label='Predicted Median TTT', color='crimson', lw=2)
        axs[0, 1].fill_between(time_steps, pred_p10, pred_p90, color='crimson', alpha=0.25, label='80% Confidence Interval')
        axs[0, 1].set_title('Time-to-Tip Horizon Tracking')
        axs[0, 1].set_xlabel('Simulation Progress (Years)')
        axs[0, 1].set_ylabel('Years Remaining Until Collapse')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # --- Plot C: Residual Tracking over Time ---
        axs[1, 0].plot(time_steps, residuals, color='purple', lw=2, label='Residual (Pred - True)')
        axs[1, 0].axhline(0, color='black', linestyle='--', alpha=0.7)
        axs[1, 0].set_title('Prediction Error (Residuals) Over Sequence Timeline')
        axs[1, 0].set_xlabel('Simulation Progress (Years)')
        axs[1, 0].set_ylabel('Error (Years)')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        # --- Plot D: Residual Histogram (Bias Check) ---
        axs[1, 1].hist(residuals, bins=15, color='seagreen', edgecolor='black', alpha=0.7)
        axs[1, 1].axvline(0, color='black', linestyle='--', alpha=0.7)
        axs[1, 1].set_title('Distribution of Validation Errors')
        axs[1, 1].set_xlabel('Error (Years)')
        axs[1, 1].set_ylabel('Frequency Count')
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return (run_inference_and_plot,)


@app.cell
def _(Dataset, glob, os, pd, torch):
    class MultiMemberOceanDataset(Dataset):
        def __init__(self, search_pattern):

            all_x, all_y, all_u = [], [], []

                # 1. Find all matching files (e.g., ensemble_01.csv, ensemble_02.csv, etc.)
            search_pattern = os.path.join("/home/users/thopri/PROMOTE/StateBased", search_pattern)
            file_paths = glob.glob(search_pattern)
            # 2. Defensive check to prevent empty torch.cat crashes
            if len(file_paths) == 0:
                raise FileNotFoundError(
                    f"No files found matching pattern '{search_pattern}' in directory '{os.getcwd()}'"
                )

            for path in file_paths:
                df = pd.read_csv(path)

                # Extract features and targets
                x = torch.tensor(df[['SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var']].values, dtype=torch.float32)
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

    return (MultiMemberOceanDataset,)


@app.cell
def _(calculate_tipping_targets_from_mld, glob, np, os, pd, xr):
    def extract_single_point_features_for_all_ensembles(base_dir):
        # --- 1. CONFIGURATION PARAMETERS ---
        # We will determine the exact (y, x) indices dynamically below!
        TARGET_LAT = 60.819
        TARGET_LON = -56.506

        ensemble_members = range(1, 41)

        var_mappings = {
            'temp': '*votemper.nc',
            'salinity': '*vosaline.nc',
            'ssh': '*zossq.nc'
        }

        # --- 2. DYNAMICALLY FIND CHOSEN PIXEL INDEX (Y, X) ---
        # We open a single target file from Member 1 to find the closest grid matrix cell
        sample_dir = os.path.join(base_dir, "HIST2", "1", "OCN", "yearly", "1950")
        sample_file = sorted(glob.glob(os.path.join(sample_dir, var_mappings['ssh'])))[0]

        with xr.open_dataset(sample_file) as ds_sample:
            # Look for typical curvilinear coordinate labels in NEMO grids
            lat_var = 'nav_lat' if 'nav_lat' in ds_sample else ('latitude' if 'latitude' in ds_sample else None)
            lon_var = 'nav_lon' if 'nav_lon' in ds_sample else ('longitude' if 'longitude' in ds_sample else None)

            if lat_var is None or lon_var is None:
                # Fallback if names are unique: look for variables containing 'lat'/'lon'
                lat_var = [v for v in ds_sample.variables if 'lat' in v][0]
                lon_var = [v for v in ds_sample.variables if 'lon' in v][0]

            print(f"Detected curvilinear coordinate arrays: Lat='{lat_var}', Lon='{lon_var}'")

            # Calculate absolute horizontal Euclidean distance to target point
            distance = np.sqrt((ds_sample[lat_var].values - TARGET_LAT)**2 + 
                               (ds_sample[lon_var].values - TARGET_LON)**2)

            # Find the 2D matrix index position of the minimum distance entry
            target_y, target_x = np.unravel_index(np.argmin(distance), distance.shape)
            print(f"Target location maps precisely to Grid Index Coordinates -> y: {target_y}, x: {target_x}")

        # --- 3. LOOP OVER ALL 40 ENSEMBLE MEMBERS ---
        for ens in ensemble_members:
            print(f"\n==========================================")
            print(f"Processing Ensemble Member: {ens:02d}")
            print(f"==========================================")

            annual_point_datasets = []

            timeline = []
            for y in range(1950, 2015):
                timeline.append(('HIST2', y))
            for y in range(2015, 2101):
                timeline.append(('SSP370', y))

            print("Stitching variables and calculating spatial gradients year-by-year...")

            for experiment, year in timeline:
                year_dir = os.path.join(base_dir, experiment, str(ens), "OCN", "yearly", str(year))

                files = {}
                for var_name, pattern in var_mappings.items():
                    found_files = sorted(glob.glob(os.path.join(year_dir, pattern)))
                    if found_files:
                        files[var_name] = found_files[0]

                if len(files) < 3:
                    continue

                ds_temp = xr.open_dataset(files['temp'])
                ds_salt = xr.open_dataset(files['salinity'])
                ds_ssh = xr.open_dataset(files['ssh'])

                # Use appropriate vertical coordinate name
                depth_dim = 'depth' if 'depth' in ds_temp.dims else ('olevel' if 'olevel' in ds_temp.dims else 'deptht')

                surf_temp = ds_temp['votemper'].isel({depth_dim: 0})
                surf_salt = ds_salt['vosaline'].isel({depth_dim: 0})
                surf_ssh = ds_ssh['zossq']

                # Spatial gradients calculated over the intact 2D planes (across 'y' and 'x' dimensions)
                # axes=(1, 2) matches dimensions (time_counter, y, x)
                ssh_grad_y, ssh_grad_x = np.gradient(surf_ssh.values, axis=(1, 2))
                ssh_grad_mag = np.sqrt(ssh_grad_y**2 + ssh_grad_x**2)

                surf_grad = xr.DataArray(
                    ssh_grad_mag, 
                    coords=surf_ssh.coords, 
                    dims=surf_ssh.dims,
                    name='SSH_Gradient'
                )

                ds_year_spatial = xr.merge([
                    surf_temp.rename('SST'),
                    surf_salt.rename('SSS'),
                    surf_grad
                ], compat='override')  # Bypasses internal redundancy checks and silences the warning

                # 4. CRITICAL FIX: Extract using direct structural matrix indexing (y, x)
                ds_year_point = ds_year_spatial.isel(y=target_y, x=target_x).compute()
                annual_point_datasets.append(ds_year_point)

                ds_temp.close()
                ds_salt.close()
                ds_ssh.close()

            if not annual_point_datasets:
                print(f"No valid data steps recovered for ensemble {ens}. Skipping.")
                continue

            print("Concatenating annual points into a unified time series...")
            # Concatenate over our discovered temporal dimension name: 'time_counter'
            pixel_ts = xr.concat(annual_point_datasets, dim='time_counter')

            # --- Calculate Dynamic Anomalies & Rolling Volatility ---
            print("Computing rolling metrics and climate anomalies...")

            # Handle time slicing dynamically regardless of dimension name mapping
            pixel_ts = pixel_ts.rename({'time_counter': 'time'})

            baseline = pixel_ts.sel(time=slice("1950", "1980"))
            climatology = baseline.groupby("time.month").mean("time")

            sst_anom = pixel_ts["SST"].groupby("time.month") - climatology["SST"]
            sss_anom = pixel_ts["SSS"].groupby("time.month") - climatology["SSS"]
            sst_rolling_var = sst_anom.rolling(time=60, center=True, min_periods=12).std()

            features_ds = xr.Dataset({
                "SST_Anom": sst_anom,
                "SSS_Anom": sss_anom,
                "SSH_Gradient": pixel_ts["SSH_Gradient"],
                "SST_Rolling_Var": sst_rolling_var
            })
            # --- 7. DOWNSAMPLE TO ANNUAL MEANS & WRITE TO CSV ---
            print("resampling data to write out to a CSV file...")
            annual_features = features_ds.resample(time="1YS").mean()
            annual_features = annual_features.bfill(dim="time").ffill(dim="time")
        
            # Extract the raw integer year directly from cftime objects
            years_vector = [t.year for t in annual_features['time'].values]
        
            # Convert to Pandas format
            df_out = annual_features.to_dataframe().reset_index()
            df_out['Year'] = years_vector
        
            # Rearrange precisely to requested headers
            required_cols = ['Year', 'SST_Anom', 'SSS_Anom', 'SSH_Gradient', 'SST_Rolling_Var']
            df_final = df_out[required_cols]
            print("calculating tipping targets using MLD as diagnostic...")
            # CRITICAL FIX A: Pass 250.0 as a pure positional argument
            df_targets = calculate_tipping_targets_from_mld(base_dir,ens, target_y, target_x, 350.0)
        
            # Merge the targets into your feature dataframe matching on the 'Year' column
            df_complete = pd.merge(df_final, df_targets, on='Year')
            print("writing data to a CSV file...")
            # CRITICAL FIX B: Save the complete unified file and remove the duplicate overwrite
            csv_filename = f"ensemble_member_{ens:02d}.csv"
            df_complete.to_csv(csv_filename, index=False)
            print(f"Successfully generated: {csv_filename}")



    return (extract_single_point_features_for_all_ensembles,)


@app.cell
def _(glob, np, os, pd, xr):
    def calculate_tipping_targets_from_mld(base_dir,ens_number, target_y, target_x, threshold_meters=500.0):
        """
        Analyzes the somxl010 variable to find the permanent convective collapse year,
        and returns annual arrays for Time_To_Tip (Y) and Event_Mask (U).
        """
    
        # 1. Gather all MLD files chronologically
        mld_files = []
        for year in range(1950, 2015):
            files = sorted(glob.glob(os.path.join(base_dir, "HIST2", str(ens_number), "OCN", "yearly", str(year), "*somxl010.nc")))
            if files: mld_files.append(files[0])
        for year in range(2015, 2101):
            files = sorted(glob.glob(os.path.join(base_dir, "SSP370", str(ens_number), "OCN", "yearly", str(year), "*somxl010.nc")))
            if files: mld_files.append(files[0])

        # 2. Extract the monthly time series for our single pixel
        print(f"Extracting MLD profile for Ensemble {ens_number:02d}...")
        mld_ds = xr.open_mfdataset(mld_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')  # Bypasses internal redundancy checks and silences the warning
        mld_pixel = mld_ds['somxl010'].isel(y=target_y, x=target_x).compute()
        mld_ds.close()

        # 3. Downsample to get the MAXIMUM winter mixing depth per calendar year
        # Deep convection in the North Atlantic happens in late winter (Jan-Feb-March).
        # '1AS' resamples to annual blocks, and .max() captures how deep the chimney opened that winter.
        annual_max_mld = mld_pixel.resample(time_counter="1YS").max()
    
        years = [t.year for t in annual_max_mld['time_counter'].values]
        mld_values = annual_max_mld.values # 1D array of max winter depths from 1950 to 2100

        # 4. FIND THE TIPPING POINT (Permanent Shoaling)
        tipping_year = None
    
        for idx, year in enumerate(years):
            # Check if the MLD is currently shallower than our threshold
            if mld_values[idx] < threshold_meters:
                # Crucial 'Permanent' check: Look ahead at ALL remaining years up to 2100.
                # If it NEVER drops back down below the threshold, this idx is the true collapse start.
                if np.all(mld_values[idx:] < threshold_meters):
                    tipping_year = year
                    break

        # 5. CONSTRUCT THE COUNTDOWN TENSORS (Y and U)
        time_to_tip = []
        event_mask = []
    
        if tipping_year is not None:
            print(f"--> Success: Permanent convective collapse detected in Year: {tipping_year}")
            for year in years:
                if year <= tipping_year:
                    time_to_tip.append(tipping_year - year) # Linear countdown
                    event_mask.append(1)                    # Uncensored (Event observed)
                else:
                    time_to_tip.append(0)                   # Post-collapse era
                    event_mask.append(1)
        else:
            # Right-Censored Case: If an ensemble member doesn't tip before 2100
            print("--> Notice: No permanent collapse detected before 2100. Marking as Right-Censored.")
            max_year = max(years)
            for year in years:
                time_to_tip.append(max_year - year) # Distance to end of observation window
                event_mask.append(0)                # Censored (We don't know when/if it will tip)

        # Return a clean dictionary to append to your existing feature extraction script
        return pd.DataFrame({
            'Year': years,
            'Max_Winter_MLD': mld_values,
            'Time_To_Tip': time_to_tip,
            'Event_Mask': event_mask
        })

    return (calculate_tipping_targets_from_mld,)


@app.cell
def _(
    DataLoader,
    MultiMemberOceanDataset,
    TemporalWeibullRegressor,
    TensorDataset,
    extract_single_point_features_for_all_ensembles,
    random_split,
    run_inference_and_plot,
    torch,
    train_weibull_tcn,
):
    # --- Execution Mock Setup ---
    if __name__ == "__main__":
        extract_from_CANARI = False
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        base_dir = "/gws/ssde/j25b/canari/shared/large-ensemble/priority"

        if extract_from_CANARI:
            extract_single_point_features_for_all_ensembles(base_dir=base_dir)


        data = MultiMemberOceanDataset("ensemble_member_*.csv")

        full_dataset = TensorDataset(data.X, data.Y, data.U)

        # 2. Define your split allocations (e.g., 80% train, 20% validation)
        train_size = 32
        val_size = 8
        num_features = 4

        # Perform the deterministic partition
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42) # Fixed seed for reproducible splits
        )

        # 3. Create your loaders
        # For training, shuffle=True mixes up the ENSEMBLE MEMBERS per batch, 
        # but completely preserves the chronological 1950 -> 2100 order inside each sequence.
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        # 2. Instantiate TCN Model
        model = TemporalWeibullRegressor(num_inputs=num_features, num_channels=[32, 64, 128])

        # 3. Train the model
        train_hist, val_hist = train_weibull_tcn(model, train_loader, val_loader, epochs=15, lr=1e-3, device=device)

        # 4. Generate your diagnostic dashboard with the actual metrics
        run_inference_and_plot(model, val_loader, train_hist, val_hist, device=device)
    return


if __name__ == "__main__":
    app.run()
