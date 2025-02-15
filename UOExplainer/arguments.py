import argparse

parser = argparse.ArgumentParser(description="Train a model with orbit basis and score learning")

# Dataset selection
parser.add_argument('--dataset', type=str, default='bac',
                    choices=['bac', 'bashapes', 'tree_cycle', 'tree_grid', 'ppi0', 'ppi1', 'ppi2', 'ppi3', 'ppi4', 'ppi5'],
                    help='Select the dataset for training. Available options: '
                         'bac' 
                         'bashapes' 
                         'tree_cycle' 
                         'tree_grid' 
                         'ppi0 to ppi5')

# Orbit Basis Learning Parameters
parser.add_argument('--o_batch', type=int, default=64,
                    help='Batch size for orbit basis learning')
parser.add_argument('--o_epochs', type=int, default=5000,
                    help='Number of training epochs for orbit basis learning')
parser.add_argument('--o_lr', type=float, default=0.01,
                    help='Learning rate for orbit basis learning')

# Score Learning Parameters
parser.add_argument('--n_concept', type=int, default=3,
                    help='Number of concepts for score learning (default: 3)')
parser.add_argument('--s_epochs', type=int, default=500,
                    help='Number of epochs for score learning')
parser.add_argument('--s_lr', type=float, default=0.003,
                    help='Learning rate for score learning')

# Additional Training Parameters
parser.add_argument('--except_class', type=list, default=[0, 4],
                    help='List of class indices to be excluded from training')
parser.add_argument('--n_sample', type=int, default=50,
                    help='Number of samples to use in explanation')

args = parser.parse_args()