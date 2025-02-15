import os
for i in range(5):
    os.system(f'python main.py --dataset bashapes --gnn gin --sparsity 0.7 --do_plot 0 --linear_search 1')
    os.system(f' python main.py --dataset bac --gnn gin --sparsity 0.7 --do_plot 0 --linear_search 1')
    os.system(f' python main.py --dataset tree_grid --gnn gin --sparsity 0.7 --do_plot 0 --linear_search 1')
    os.system(f' python main.py --dataset tree_cycle --gnn gin --sparsity 0.7 --do_plot 0 --linear_search 1')