import pandas as pd
import numpy as np

# Set seed for reproducible climate noise
np.random.seed(42)
ensemble_num = 40
years = np.arange(1950, 2101)
tipping_year = 2045

for i in range(ensemble_num):

    sst_anom = []
    sss_anom = []
    ssh_grad = []
    sst_var = []
    time_to_tip = []
    event_mask = []
    
    for y in years:
        # 1. Calculate target variables
        mask = 1
        if y <= tipping_year:
            ttt = tipping_year - y
        else:
            ttt = 0 # Event has already occurred
            
        time_to_tip.append(ttt)
        event_mask.append(mask)
        
        # 2. Synthesize physical state dynamics based on timeline phase
        noise = np.random.normal(0, 0.05)
        
        if y < 2010:
            # Stable baseline era
            sst_anom.append(0.0 + (y - 1950) * 0.004 + noise)
            sss_anom.append(0.0 - (y - 1950) * 0.001 + noise * 0.5)
            ssh_grad.append(0.45 - (y - 1950) * 0.001 + noise)
            sst_var.append(0.015 + np.random.uniform(-0.002, 0.002))
            
        elif 2010 <= y < tipping_year:
            # Critical Slowing Down / Progressive Stratification Era
            pct = (y - 2010) / (tipping_year - 2010)
            sst_anom.append(0.25 + pct * 0.7 + noise * 1.5) # Increased volatility
            sss_anom.append(-0.08 - pct * 0.4 + noise * 0.5) # Fast freshening
            ssh_grad.append(0.38 - pct * 0.3 + noise) # Circulation slowing down
            sst_var.append(0.022 + pct * 0.13 + np.random.uniform(-0.01, 0.01)) # Spiking variance
            
        else:
            # Post-Collapse Regime (Abrupt drop due to convective termination)
            sst_anom.append(-1.0 + noise) # Cold anomaly over the SPG
            sss_anom.append(-0.75 + noise * 0.3) # Permanent fresh state
            ssh_grad.append(-0.2 + noise) # Weakened/reversed subpolar circulation
            sst_var.append(0.03 + np.random.uniform(-0.005, 0.005))
    
    # Assemble into Pandas DataFrame
    df = pd.DataFrame({
        'Year': years,
        'SST_Anom': np.round(sst_anom, 3),
        'SSS_Anom': np.round(sss_anom, 3),
        'SSH_Gradient': np.round(ssh_grad, 3),
        'SST_Rolling_Var': np.round(sst_var, 4),
        'Time_To_Tip': time_to_tip,
        'Event_Mask': event_mask
    })
    
    # Save to file
    df.to_csv(f'ensemble_member_0{i+1}.csv', index=False)
    print(f"Successfully generated 'ensemble_member_0{i+1}.csv' containing 151 sequence steps.")