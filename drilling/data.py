import numpy as np
import pandas as pd
import logging
from colorama import Fore, Style

logger = logging.getLogger("DrillGPT")

def generate_synthetic_drilling_data(n_samples=500):
    """Generate synthetic drilling data with realistic relationships."""
    logger.info(Fore.LIGHTMAGENTA_EX + f"Generating {n_samples} synthetic drilling data points..." + Style.RESET_ALL)
    # Define formations and bit types
    formation_types = ['Sandstone', 'Shale', 'Limestone', 'Dolomite']
    bit_types = ['PDC', 'Tricone', 'Diamond']
    
    # Generate base parameters
    data = {
        'depth': np.cumsum(np.random.uniform(5, 30, n_samples)),
        'formation': np.random.choice(formation_types, n_samples),
        'bit_type': np.random.choice(bit_types, n_samples),
        'weight_on_bit': np.random.uniform(5, 30, n_samples),  # kips
        'rotary_speed': np.random.uniform(40, 200, n_samples),  # RPM
        'flow_rate': np.random.uniform(300, 1000, n_samples),  # GPM
        'bit_hours': np.zeros(n_samples),  # Hours on bit
    }
    
    df = pd.DataFrame(data)
    
    # Add formation hardness
    hardness_map = {'Sandstone': 5000, 'Shale': 3000, 'Limestone': 8000, 'Dolomite': 10000}
    df['formation_hardness'] = df['formation'].map(hardness_map)
    
    # Calculate bit hours
    for i in range(1, n_samples):
        if df.loc[i, 'bit_type'] != df.loc[i-1, 'bit_type']:
            df.loc[i, 'bit_hours'] = 0
        else:
            df.loc[i, 'bit_hours'] = df.loc[i-1, 'bit_hours'] + np.random.uniform(0.5, 2.0)
    
    # Calculate rate of penetration (ROP)
    # Base ROP with dependencies
    base_rop = 50 * np.random.normal(1, 0.2, n_samples)
    hardness_factor = 10000 / (df['formation_hardness'] + 1000)
    wob_factor = 0.8 + 0.4 * np.tanh((df['weight_on_bit'] - 15) / 5)
    rpm_factor = 0.6 + 0.8 * np.tanh((df['rotary_speed'] - 120) / 30)
    flow_factor = 0.7 + 0.6 * np.tanh((df['flow_rate'] - 650) / 150)
    bit_wear_factor = 1.2 - 0.5 * np.tanh((df['bit_hours'] - 50) / 20)
    
    df['rop'] = base_rop * hardness_factor * wob_factor * rpm_factor * flow_factor * bit_wear_factor
    df['rop'] = df['rop'].clip(5, 150)  # Reasonable bounds
    
    # Add vibration data
    df['vibration'] = 0.2 + 0.2 * np.random.normal(0, 1, n_samples) + 0.02 * df['weight_on_bit'] + 0.001 * df['rotary_speed']
    df['vibration'] = df['vibration'].clip(0, None)
    
    # Calculate bit failure probabilities
    failure_prob = 0.2 * np.tanh((df['bit_hours'] - 60) / 15) + 0.3 * np.tanh((df['vibration'] - 1.5) / 0.5)
    failure_prob = 0.2 + 0.8 * (failure_prob - failure_prob.min()) / (failure_prob.max() - failure_prob.min())
    df['bit_failure'] = (np.random.random(n_samples) < failure_prob).astype(int)
    
    return df 