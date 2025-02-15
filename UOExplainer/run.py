from arguments import args
from utils import *
import torch_geometric
import torch
from explainer import Explainer
from torch.utils.data import DataLoader
import sys

if __name__ == "__main__":
    data_name = args.data_name
    feature, label, edge = load_data(data_name)

    #data = torch_geometric.data.Data(x=feature, edge_index=edge, y=label)

    model, model_path = load_model(data_name)
    model.eval()

    pred = (model(feature, edge) - model.ffn[1].bias).detach()

    predict = torch.argmax(torch.softmax(model(feature, edge), dim=1), dim=1)
    answer = (predict == label).int()
    print(f"Pretrained GNN ACC: {torch.mean(answer.float()).item()}")


    answer_idx = torch.flatten(torch.argwhere(answer == 1))

    #answer = (predict == label)
    #print(answer.shape)

    explainer = Explainer(data_name = data_name, model = model, model_path = model_path, answer_idx = answer_idx, pred= pred)
    explainer.orbit_basis_learning()
    explainer.score_learning()
    if args.data_name == 'bashapes':
        nodes = range(400, 700, 6)
    if args.data_name == 'bac':
        nodes = range(400, 1400, 11)
    if args.data_name == 'tree_cycle':
        nodes = [i for i in range(0, 870, 5)]
    if args.data_name == 'tree_grid':
        nodes = [i for i in range(0, 1230, 10)]
    models = explainer.analyze(nodes)
    models = ' '.join(map(str, models))
    result_txt = open('../result.txt', 'w')
    result_txt.write(models)
    print(models)
#hidden_d = model.ffn[-1].weight.shape[1]




