# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import get_analyzer, find_files_prefix, find_files_suffix
from arguments import parser
import datetime

args = parser.parse_args()

if args.use_sns:
    import seaborn as sns
    sns.set()

def extract_params_from_path(path):
    """Extract training parameters from tensorboard path.
    
    Args:
        path (str): Tensorboard path containing parameter information
        
    Returns:
        dict: Dictionary containing zero_stage, tp, pp, dp, sp, and mbsz values
    """
    # Extract zero stage
    zero_stage = '1'  # default
    for z in ['z1', 'z2', 'z3']:
        if z in path:
            zero_stage = z[1]
            break
            
    # Extract parallelism settings
    tp = path.split('tp')[1].split('_')[0]
    pp = path.split('pp')[1].split('_')[0]
    dp = path.split('dp')[1].split('_')[0]
    sp = path.split('sp')[1].split('_')[0]
    mbsz = path.split('mbsz')[1].split('_')[0]
    
    return {
        'zero_stage': zero_stage,
        'tp': tp,
        'pp': pp,
        'dp': dp,
        'sp': sp,
        'mbsz': mbsz
    }

def get_plot_name(base_name, params):
    """Generate plot name with parameters.
    
    Args:
        base_name (str): Original plot name
        params (dict): Dictionary of parameters
        
    Returns:
        str: Plot name with parameters
    """
    base, ext = os.path.splitext(base_name)
    return f"{base}_z{params['zero_stage']}_tp{params['tp']}_pp{params['pp']}_dp{params['dp']}_sp{params['sp']}_mbsz{params['mbsz']}{ext}"

def main():
    # Create dated output directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    target_prefix = 'events.out.tfevents'
    tb_log_paths = find_files_prefix(args.tb_dir, target_prefix)
    print(f"Found {len(tb_log_paths)} matching files")
    print(tb_log_paths)
    analyzer = get_analyzer(args.analyzer)

    for tb_path in tb_log_paths:
        print(f"Processing: {tb_path}")
        try:
            analyzer.set_names(tb_path)

            event_accumulator = EventAccumulator(tb_path)
            event_accumulator.Reload()

            events = event_accumulator.Scalars(args.tb_event_key)

            x = [x.step for x in events]
            y = [x.value for x in events]

            label = analyzer.get_label_name()
            params = extract_params_from_path(tb_path)
            label = f'{label}, MBSZ={params["mbsz"]}'

            plt.plot(x, y, label=label)

            if not args.skip_csv:
                df = pd.DataFrame({"step": x, "value": y})
                csv_filename = os.path.join(output_dir, 
                    f"{args.csv_name}{analyzer.get_csv_filename()}_z{params['zero_stage']}_tp{params['tp']}_pp{params['pp']}_dp{params['dp']}_sp{params['sp']}_mbsz{params['mbsz']}.csv")
                df.to_csv(csv_filename)
        except Exception as e:
            print(f"Error processing {tb_path}: {str(e)}")
            continue

    plt.grid(True)

    if not args.skip_plot:
        plt.legend()
        plt.title(args.plot_title)
        plt.xlabel(args.plot_x_label)
        plt.ylabel(args.plot_y_label)
        params = extract_params_from_path(tb_path)
        plot_name = get_plot_name(args.plot_name, params)
        plt.savefig(os.path.join(output_dir, plot_name))

def plot_csv():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", current_time)
    os.makedirs(output_dir, exist_ok=True)
    
    target_suffix = 'csv'
    csv_log_files = find_files_suffix(args.csv_dir, target_suffix)

    analyzer = get_analyzer(args.analyzer)

    for csv_file in csv_log_files:
        analyzer.set_names(csv_file)

        x, y = [], []
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] == 'step':
                    continue
                x.append(int(row[1]))
                y.append(float(row[2]))

        plt.plot(x, y, label=f'{analyzer.get_label_name()}')

    plt.grid(True)
    plt.legend()
    plt.title(args.plot_title)
    plt.xlabel(args.plot_x_label)
    plt.ylabel(args.plot_y_label)
    plt.savefig(os.path.join(output_dir, args.plot_name))

if __name__ == "__main__":
    if args.plot_only:
        plot_csv()
    else:
        main()
