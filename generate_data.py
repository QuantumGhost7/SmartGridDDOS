import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_network_traffic(n_samples, filename_prefix):
    # Calculate number of samples for each class
    normal_samples = n_samples // 2
    ddos_samples = n_samples - normal_samples
    
    # Generate timestamps
    base_timestamp = datetime.now()
    timestamps = [base_timestamp + timedelta(seconds=x) for x in range(n_samples)]
    
    # Generate normal traffic data
    normal_data = {
        'packets_per_second': np.random.normal(500, 200, normal_samples),
        'speed_mbps': np.random.normal(1000, 300, normal_samples),
        'packet_size': np.random.normal(500, 100, normal_samples),
        'is_ddos': np.zeros(normal_samples)
    }
    
    # Generate DDoS traffic data
    ddos_data = {
        'packets_per_second': np.random.normal(4000, 800, ddos_samples),
        'speed_mbps': np.random.normal(5000, 1000, ddos_samples),
        'packet_size': np.random.normal(300, 50, ddos_samples),
        'is_ddos': np.ones(ddos_samples)
    }
    
    # Combine normal and DDoS data
    combined_data = {
        'tsimestamp': timestamps,
        'packets_per_second': np.concatenate([normal_data['packets_per_second'], ddos_data['packets_per_second']]),
        'speed_mbps': np.concatenate([normal_data['speed_mbps'], ddos_data['speed_mbps']]),
        'packet_size': np.concatenate([normal_data['packet_size'], ddos_data['packet_size']]),
        'is_ddos': np.concatenate([normal_data['is_ddos'], ddos_data['is_ddos']])
    }
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Ensure no negative values
    df['packets_per_second'] = df['packets_per_second'].clip(lower=0)
    df['speed_mbps'] = df['speed_mbps'].clip(lower=0)
    df['packet_size'] = df['packet_size'].clip(lower=64)  # Minimum packet size
    
    # Add some noise
    df['packets_per_second'] += np.random.normal(0, 50, len(df))
    df['speed_mbps'] += np.random.normal(0, 100, len(df))
    df['packet_size'] += np.random.normal(0, 25, len(df))
    
    # Clip values again after adding noise
    df['packets_per_second'] = df['packets_per_second'].clip(lower=0)
    df['speed_mbps'] = df['speed_mbps'].clip(lower=0)
    df['packet_size'] = df['packet_size'].clip(lower=64)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the dataset
    df.to_csv(f'{filename_prefix}_data.csv', index=False)
    print(f"Generated {len(df)} samples with {df['is_ddos'].sum()} DDoS events")
    return df

if __name__ == "__main__":
    # Generate training data
    print("Generating training data...")
    train_df = generate_network_traffic(10000, 'train')
    
    # Generate separate testing data
    print("\nGenerating testing data...")
    test_df = generate_network_traffic(2000, 'test')
    
    print("\nDatasets generated successfully!")