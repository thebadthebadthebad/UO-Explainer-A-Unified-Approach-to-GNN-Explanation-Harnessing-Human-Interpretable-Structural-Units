
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from GNN_Models.gin import GIN, GIN_NC
from GNN_Models.gcn import GCN, GCN_NC
from GNN_Models.graph_sage import GraphSAGE
from torch.utils.data import random_split
from itertools import combinations
from collections import deque
import random
import sys
import os

sys.path.append('/home/steve305/kr/mmm/other_methods/EiG-Search/Utils/models/')

def check_task(dataname):
    if dataname in ["ba_shape", "tree_grid", "ba_community"]:
        return "NC"
    else: return "GC"

def load_model(dataname, gnn, n_fea, n_cls):
    if dataname in ["ba_shape"]:
        if gnn == "gin":
            model = GIN_NC(n_fea, n_cls, 3, 32).cuda()
        elif gnn == "gcn":
            model = GCN_NC(n_fea, n_cls, 3, 32).cuda()
    elif dataname in ["tree_grid", "ba_community"]:
        model = GIN_NC(n_fea, n_cls, 3, 64).cuda()
    elif dataname in ["ba_2motifs"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 32).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 32).cuda()
        elif gnn == "sage":
            model = GraphSAGE(n_fea, n_cls, 3, 32).cuda()
    elif dataname in ["MUTAG", "bbbp"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 128).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 128).cuda()
    elif dataname in ["Mutagenicity_nh2no2"]:
        model = GIN(n_fea, n_cls, 2, 32).cuda()
    elif dataname in ["Mutagenicity"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "sage":
            model = GraphSAGE(n_fea, n_cls, 3, 64).cuda()
    elif dataname in ["NCI1"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "sage":
            model = GraphSAGE(n_fea, n_cls, 3, 64).cuda()
    elif dataname in ["Graph-SST2"]:
        model = GCN(n_fea, n_cls, 3, 128).cuda()

    model.load_state_dict(torch.load("saved_models/"+gnn+'_'+dataname+".model"))
    return model

def my_model_selector(data):
    data_path = f"/home/steve305/kr/mmm/dataset/{data}/"
    model_path = [i for i in os.listdir(data_path) if i.endswith('00')]
    if len(model_path) == 1:
        model_path = f'{data_path}{model_path[0]}/'
    model = torch.load(f'{model_path}model.pth').cuda()

    return model



def my_ground_truth_loader(dataset):
    feature, label, edge = my_load_data(dataset)
    ground_truth = []
    graph = make_graph(dataset)
    if dataset == 'bashapes':
        indices = range(400, 700, 6)
        for node in indices:
            gt = find_baco_gt(graph, node, label)
            ground_truth.append(gt)

    if dataset == 'bac':
        indices = [i for i in range(400, 1400, 11)]
        unuse = []
        for node in indices:
            if label[node] == 4:
                unuse.append(node)
                continue
            gt = find_baco_gt(graph, node, label)
            ground_truth.append(gt)
        for i in unuse:
            indices.remove(i)

    if dataset == 'tree_cycle':
        indices = [i for i in range(0, 870, 5)]
        unuse = []

        for node in indices:
            if label[node] == 0:
                unuse.append(node)
                continue
            gt = find_76_orbit(graph, node)
            ground_truth.append(gt)
        for i in unuse:
            indices.remove(i)

    if dataset == 'tree_grid':
        indices = [i for i in range(0, 1230, 10)]
        unuse = []
        for node in indices:
            if label[node] == 0:
                unuse.append(node)
                continue
            if label[node] == 1:
                gt = find_73_orbit(graph, node)
            elif label[node] == 2:
                gt = find_74_orbit(graph, node)
            else:
                gt = find_75_orbit(graph, node)
            ground_truth.append(gt)
        for i in unuse:
            indices.remove(i)


    return ground_truth, indices

def detect_exp_setting(dataname, dataset):
    if dataname in ["ba_2motifs"]:
        return range(len(dataset))
    elif dataname in ["ba_shape"]:
        return range(300, dataset.x.shape[0])
    elif dataname in ["ba_community"]:
        return list(range(300, 700))+list(range(1000, 1400))
    elif dataname in ["tree_grid"]:
        return range(511, dataset.x.shape[0])
        # return range(511, 800)
    elif dataname in ["MUTAG", "Mutagenicity_nh2no2", "NCI1"]:
        return torch.where(dataset.data.y==1)[0].tolist()
    elif dataname in ["bbbp", "Mutagenicity"]:
        return torch.where(dataset.data.y==0)[0].tolist()
    elif dataname == "Graph-SST2":
        # print(dir(dataset))
        # print(dataset._indices)
        # exit(0)
        num_train = int(0.8 * len(dataset))
        num_eval = int(0.1 * len(dataset))
        num_test = len(dataset) - num_train - num_eval
        _,_, test_dataset = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                            generator=torch.Generator().manual_seed(1234))
        return list(set(test_dataset.indices).intersection(set(torch.where(dataset.data.y==1)[0].tolist())))[:500]
        

def detect_motif_nodes(dataname):
    if dataname in ["ba_shape", "ba_2motifs", "ba_community"]:
        return 5
    elif dataname in ["tree_grid"]:
        return 9
    return None

def show():
    plt.show()
    plt.clf()

def GC_vis_graph(x, edge_index, Hedges=None, good_nodes=None, datasetname=None, edge_color='red'):
    colors = [  'orange',   'red',      'lime',         'green',
                'blue',     'orchid',   'darksalmon',   'darkslategray',
                'gold',     'bisque',   'tan',          'lightseagreen',
                'indigo',   'navy',     'aliceblue',     'violet', 
                'palegreen', 'lightsalmon', 'olive', 'peru',
                'cyan']
    edges = np.transpose(np.asarray(edge_index.cpu()))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if datasetname not in ["bbbp"]:
        node_label = list(np.asarray(torch.argmax(x, dim=1).cpu()))
    else: 
        node_label = list(np.asarray(x[:,0].cpu()))
    
    max_label = max(node_label) + 1
    nmb_nodes = len(node_label)

    pos = nx.kamada_kawai_layout(G)

    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        if i in G.nodes():
            label2nodes[node_label[i]].append(i)

    for i in range(max_label):
        node_filter = []
        for j in range(len(label2nodes[i])):
            node_filter.append(label2nodes[i][j])
        if i < len(colors): cc = colors[i]
        else: cc = colors[-1]
        if edge_color=='red':
            nx.draw_networkx_nodes(G, pos,
                                nodelist=node_filter,
                                node_color=cc,
                                node_size=100)
    
    if edge_color=='red':
        nx.draw_networkx_labels(G, pos, {o:o for o in list(pos.keys())})
        nx.draw_networkx_edges(G, pos, width=2,  edge_color='grey', arrows=False)

    if Hedges is None:
        edges = np.transpose(np.asarray(edge_index.cpu()))
        _edges = np.min(sum(np.asarray([edges == nd for nd in good_nodes])), axis=-1)
        Hedges = np.nonzero(_edges)[0]
    elif good_nodes is None: 
        good_nodes = list(set(edge_index[0,Hedges].tolist() + edge_index[1,Hedges].tolist()))

    nx.draw_networkx_edges(G, pos,
                                edges[Hedges],
                                width=5, edge_color=edge_color, arrows=True)

    plt.axis('off')
    # plt.clf()

def NC_vis_graph(edge_index, y, node_idx, datasetname=None, un_edge_index=None, nodelist=None, H_edges=None):
    y = torch.argmax(y, dim=-1)
    node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
    node_color = ['#FFA500', '#4970C6', '#FE0000', 'green','orchid','darksalmon','darkslategray','gold','bisque','tan','navy','indigo','lime',]
    colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    node_idx = int(node_idx)
    edges = np.transpose(np.asarray(edge_index.cpu()))
    if un_edge_index is not None: un_edges = np.transpose(np.asarray(un_edge_index.cpu()))
    if nodelist is not None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in edges if
                                n_frm in nodelist and n_to in nodelist]
        nodelist = nodelist.tolist()
    elif H_edges is not None:
        if un_edge_index is not None: edgelist = un_edges[H_edges]
        else: edgelist = edges[H_edges]
        nodelist = list(set(list(edgelist[:,0])+list(edgelist[:,1])))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if datasetname == "tree_grid":
        G = nx.ego_graph(G, node_idx, radius=3)
    elif datasetname == "ba_community":
        G = nx.ego_graph(G, node_idx, radius=2)
    else:
        G = nx.ego_graph(G, node_idx, radius=3)

    for n in nodelist:
        if n not in G.nodes:
            nodelist.remove(n)
    def remove_unavail(edgelist):
        for i, tup in enumerate(edgelist):
            if tup[0] not in G.nodes or tup[1] not in G.nodes:
                edgelist = np.delete(edgelist, i, axis=0)
                return edgelist, i
        return edgelist, len(edgelist)
    edgelist, i = remove_unavail(edgelist)
    while i != len(edgelist):
        edgelist, i = remove_unavail(edgelist)

    pos = nx.kamada_kawai_layout(G) # calculate according to graph.nodes()
    pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
    colors = [colors[pp] for pp in list(G.nodes)]

    nx.draw_networkx_nodes(G, pos,
                            nodelist=list(G.nodes()),
                            node_color=colors,
                            node_size=100)
    if isinstance(colors, list):
        list_indices = int(np.where(np.array(G.nodes()) == node_idx)[0])
        node_idx_color = colors[list_indices]
    else:
        node_idx_color = colors

    nx.draw_networkx_nodes(G, pos=pos,
                            nodelist=[node_idx],
                            node_color=node_idx_color,
                            node_size=400)

    nx.draw_networkx_edges(G, pos, width=1, edge_color='grey', arrows=False)
    if nodelist is not None or H_edges is not None:
        nx.draw_networkx_edges(G, pos=pos_nodelist,
                            edgelist=edgelist, width=2,
                            edge_color='red',
                            arrows=False)

    labels = {o:o for o in list(G.nodes)}
    nx.draw_networkx_labels(G, pos,labels)
    
    # plt.axis('off')

def NLP_vis_graph(x, edge_index, nodelist, words, Hedges=None, figname=None):
    edges = np.transpose(np.asarray(edge_index.cpu()))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    node_label = list(np.asarray(torch.argmax(x, dim=1).cpu()))

    pos = nx.kamada_kawai_layout(G)
    words_dict = {i: words[i] for i in G.nodes}

    # nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=300)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='grey')

    if nodelist is not None:
        pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
        nx.draw_networkx_nodes(G, pos_coalition,
                                nodelist=nodelist,
                                node_color='yellow',
                                node_shape='o',
                                node_size=600)
    if Hedges is None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in G.edges()
                    if n_frm in nodelist and n_to in nodelist]
        nx.draw_networkx_edges(G, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='red')
    else: 
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges[Hedges], width=5, edge_color='red')

    
    
    nx.draw_networkx_labels(G, pos, words_dict)

    plt.axis('off')
    plt.show()




def find_thres(scos):
    last_sc, last_diff = None, 0.0
    init=scos[0]
    for p, sc in enumerate(scos):
        if last_sc is None:
            last_sc = sc
        if p>2 and (2.1*sc < init or last_sc > 2*sc or sc < last_diff*1.5 or (last_sc >1.5 *sc and last_diff>0.0 and last_sc-sc > last_diff*3)):
        # if p>1 and (last_sc > 2*sc or sc < last_diff*1.5 or (last_sc >1.5 *sc and last_diff>0.0 and last_sc-sc > last_diff*3)):
            # print(f'sc: {sc}, last_sc: {last_sc}, lst diff:{last_diff}')
            return p
        last_diff = last_sc-sc
        last_sc = sc
    return p



def make_graph(dataset):
    edge_list = open(f"/home/steve305/kr/mmm/dataset/{dataset}/edge_list.txt", 'r')
    graph = nx.Graph()
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)

    return graph

def find_baco_gt(graph, node, label):
    if label[node] == 0 or label[node] == 4:
        return
    if label[node] in [5, 7, 6]:
        ground_dict = {5: 0, 6: 0, 7: 1}
    else:
        ground_dict = {1: 0, 2: 0, 3: 1}
    neighbor_nodes = deque([node])
    ground_nodes = []
    ground_edges = []
    while True:
        now_node = neighbor_nodes.popleft()

        now_label = label[now_node].item()

        if now_label in ground_dict.keys():
            if ground_dict[now_label] < 2:
                ground_nodes.append(now_node)
                ground_dict[now_label] += 1

                neigbor = list(graph.neighbors(now_node))

                neighbor_nodes += neigbor

                for j in neigbor:
                    if label[j].item() != 0 and label[j].item() != 4 and abs(j-now_node)<10 :
                        ground_edges.append((now_node, j))
            if sum(ground_dict.values()) == 6:
                if len(ground_edges) == 12:
                    return ground_edges

def find_76_orbit(graph, node):
    edge_matrix = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for i, j in graph.edges:
        edge_matrix[i, j] = 1
        edge_matrix[j, i] = 1
    neigbor = list(graph.neighbors(node))
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a, b] == 1:
            continue
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        b_neighbor = list(graph.neighbors(b))
        b_neighbor.remove(node)
        random.shuffle(a_neighbor)
        random.shuffle(b_neighbor)
        for c in a_neighbor:
            if edge_matrix[c, node] == 1 or  edge_matrix[c, b] == 1:
                continue
            for d in b_neighbor:
                if edge_matrix[node, d] == 1 or edge_matrix[a, d] == 1 or edge_matrix[c, d] == 1:
                    continue
                c_neighbor = list(graph.neighbors(c))
                c_neighbor.remove(a)
                random.shuffle(c_neighbor)
                for e in c_neighbor:
                    if edge_matrix[e, d] == 0 or edge_matrix[e,node] == 1 or edge_matrix[e,a]==1 or edge_matrix[e,b]==1:
                        continue
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (c, e), (e, c), (d, e), (e, d), (b, d), (d, b)]

                    n +=1
                if n>1:
                    print('fuck')
                    exit()
    return ground_edge


def find_73_orbit(graph, node):
    edge_matrix = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for i, j in graph.edges:
        edge_matrix[i, j] = 1
        edge_matrix[j, i] = 1
    neigbor = list(graph.neighbors(node))
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0

    for ida, idb, idc, idd in combination:
        combi2 = [(ida, idb, idc, idd),(idb,ida, idc,idd), (idd, idb, idc, ida) ]
        random.shuffle(combi2)
        for a, b, c, d in combi2:
            if edge_matrix[a, d] == 1 or edge_matrix[a, b] == 1 or edge_matrix[b, c] == 1 or edge_matrix[d, c] == 1 or edge_matrix[a, c] == 1 or edge_matrix[b, d] == 1:
                continue
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(node)
            c_neighbor = list(graph.neighbors(c))
            c_neighbor.remove(node)
            d_neighbor = list(graph.neighbors(d))
            d_neighbor.remove(node)
            ab_intersect = set(a_neighbor) & set(b_neighbor)
            ad_intersect = set(a_neighbor) & set(d_neighbor)
            bc_intersect = set(b_neighbor) & set(c_neighbor)
            cd_intersect = set(c_neighbor) & set(d_neighbor)
            random.shuffle(ab_intersect)
            random.shuffle(ad_intersect)
            random.shuffle(bc_intersect)
            random.shuffle(cd_intersect)
            for ab in ab_intersect:
                if edge_matrix[node, ab] == 1 or edge_matrix[ab, d] == 1 or edge_matrix[ab, c] == 1:
                    continue
                for ad in ad_intersect:
                    if edge_matrix[ad, node] == 1 or edge_matrix[ad, b] == 1 or edge_matrix[ad, c] == 1 or edge_matrix[ad, ab] == 1:
                        continue
                    for bc in bc_intersect:
                        if edge_matrix[bc, node] == 1 or edge_matrix[bc, a] == 1 or edge_matrix[bc, d] == 1 or edge_matrix[bc, ad] == 1 or edge_matrix[bc, ab] == 1:
                            continue
                        for cd in cd_intersect:
                            if edge_matrix[cd, node] == 1 or edge_matrix[cd, a] == 1 or edge_matrix[cd, b] == 1 or edge_matrix[cd, ad] == 1 or edge_matrix[cd, ab] == 1 or edge_matrix[cd, bc] == 1:
                                continue
                            ground_edge = [(node, a), (a, node), (b, node), (node, b), (node, c), (c, node), (d, node), (node, d), (b, ab), (ab, b), (b, bc),(bc, b), (a, ab),
                                           (ab, a), (a, ad), (ad, a), (ad, d), (d, ad), (d, cd), (cd, d), (bc, c), (c, bc), (c, cd), (cd, c)]

    return ground_edge

def find_74_orbit(graph, node):
    edge_matrix = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for i, j in graph.edges:
        edge_matrix[i, j] = 1
        edge_matrix[j, i] = 1
    neigbor = list(graph.neighbors(node))
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0

    for ida, idb, idc in combination:
        combi2 = [(ida, idb, idc), (idb, idc, ida), (idc, idb, ida)]
        random.shuffle(combi2)
        for a, b, c in combi2:
            if edge_matrix[b, a] == 1 or edge_matrix[b, c] == 1 or edge_matrix[c, a] == 1:
                continue
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            a_neighbor_combi = list(combinations(a_neighbor, 3))
            random.shuffle(a_neighbor_combi)
            for dbb, dcc, daa in a_neighbor_combi:
                neighbor_combi = [(daa, dbb, dcc), (daa, dcc, dbb), (dbb, daa, dcc), (dbb, dcc, daa), (dcc, daa, dbb), (dcc, dbb, daa)]
                random.shuffle(neighbor_combi)
                for aa, bb, cc in neighbor_combi:
                    if edge_matrix[bb, node] == 1 or edge_matrix[bb, cc] == 1 or edge_matrix[bb, c] == 1 or edge_matrix[bb, b] == 0 or edge_matrix[bb, aa] == 1 or edge_matrix[cc, node] == 1 or edge_matrix[cc, b] == 1 or edge_matrix[cc, c] == 0 or edge_matrix[cc, aa] == 1 or edge_matrix[b, aa] == 1 or edge_matrix[c, aa] == 1:
                        continue
                    aa_neighbor = list(graph.neighbors(aa))
                    aa_neighbor.remove(a)
                    aa_combi = list(combinations(aa_neighbor, 2))
                    random.shuffle(aa_combi)
                    for dab, dac in aa_combi:
                        if edge_matrix[dab, dac] == 1 or edge_matrix[dab, node] == 1 or edge_matrix[dac, node] == 1 or edge_matrix[dab, b] == 1 or edge_matrix[dab, c] == 1 or edge_matrix[dac, c] == 1 or edge_matrix[dac, b] == 1 or edge_matrix[dab, a] == 1 or edge_matrix[dac, a] == 1 :
                            continue
                        aa_combi_list = [(dab, dac), (dac, dab)]
                        random.shuffle(aa_combi_list)
                        for ab, ac in aa_combi_list:
                            if edge_matrix[ab, bb] == 0 or edge_matrix[ac, cc] == 0 or edge_matrix[ab, cc] == 1 or edge_matrix[ac, bb] == 1:
                                continue
                            ground_edge = [(node, a), (a, node), (b, node), (node, b), (node, c), (c, node), (b, bb), (bb, b), (c, cc), (cc, c), (a, bb), (bb, a), (a, cc), (cc, a), (bb, ab), (ab, bb),(ab, aa), (aa, ab), (a, aa), (aa, a),
                                           (cc, ac), (ac, cc), (aa, ac), (ac, aa)]
    return ground_edge

def find_75_orbit(graph, node):
    edge_matrix = torch.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for i, j in graph.edges:
        edge_matrix[i, j] = 1
        edge_matrix[j, i] = 1
    neigbor = list(graph.neighbors(node))

    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0

    for ida, idb in combination:
        if edge_matrix[ida, idb] == 1:
            continue
        combi2 = [(ida, idb)]
        random.shuffle(combi2)
        for a, b in combi2:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            a_combi = list(combinations(a_neighbor, 2))
            random.shuffle(a_combi)
            for dd, dc in a_combi:
                if edge_matrix[dd, dc] == 1 or edge_matrix[node, dd] == 1 or edge_matrix[node, dc] == 1:
                    continue
                a_combi2 = [(dd, dc), (dc, dd)]
                random.shuffle(a_combi2)
                for c, d in a_combi2:
                    if edge_matrix[c, b] == 1 or edge_matrix[b, d] == 0:
                        continue
                    d_neighbor = list(graph.neighbors(d))
                    d_neighbor.remove(a)
                    d_neighbor.remove(b)
                    d_combi = list(combinations(d_neighbor, 2))
                    random.shuffle(d_combi)
                    for de, df in d_combi:
                        if edge_matrix[de, df] == 1 or edge_matrix[node, df] == 1 or edge_matrix[de, node] == 1 or edge_matrix[de, a] == 1 or edge_matrix[a, df] == 1 or edge_matrix[b, df] == 1 or edge_matrix[de, b] == 1:
                            continue
                        d_combi2 = [(de, df), (df, de)]
                        random.shuffle(d_combi2)
                        for e, f in d_combi2:
                            if edge_matrix[c, f] == 0 or edge_matrix[e, c] == 1:
                                continue
                            e_neighbor = list(graph.neighbors(e))
                            e_neighbor.remove(d)
                            e_combi = list(combinations(e_neighbor, 2))
                            random.shuffle(e_combi)
                            for dg, dh in e_combi:
                                if edge_matrix[dg, dh] == 1 or edge_matrix[d, dg] == 1 or edge_matrix[d, dh] == 1 or edge_matrix[node, dg] == 1 or edge_matrix[node, dh] == 1 or edge_matrix[dh, c] == 1 or edge_matrix[c, dg] == 1 or edge_matrix[a, dg] == 1 or edge_matrix[dh, a] == 1:
                                    continue
                                e_combi2 = [(dg, dh), (dh, dg)]
                                random.shuffle(e_combi2)
                                for g, h in e_combi2:
                                    if edge_matrix[b, g] == 0 or edge_matrix[h, f] == 0 or edge_matrix[f, g] == 1 or edge_matrix[h, b] == 1:
                                        continue
                                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (c, f), (f, c), (a, d),(d, a), (b, d), (d, b), (d, f), (f, d), (b, g), (g, b), (g, e), (e, g), (e, h), (h, e), (f, h), (h, f), (d, e), (e, d)]

    return ground_edge

def my_load_data(data):
    data_path = f"/home/steve305/kr/mmm/dataset/{data}/"
    edge_list = open(f'{data_path}edge_list.txt', 'r')
    edge = [[], []]
    for line in edge_list:
        node1, node2 = map(int, line.strip().split(' '))
        edge[0].append(node1)
        edge[1].append(node2)
    edge = torch.LongTensor(edge)

    feature_list = open(f'{data_path}features.txt', 'r')
    feature_list = feature_list.readlines()
    feature_dim = len(feature_list[0].strip().split(' ')) -2
    n_node = len(feature_list)
    label = torch.zeros((n_node))
    feature = torch.zeros((n_node,feature_dim ))

    for features in feature_list:
        features = list(map(float, features.strip().split()))
        node, features_, label_ = features[0], features[1:-1], features[-1]

        label[int(node)] = label_
        feature[int(node), :] = torch.FloatTensor(features_)
    return feature, label.long().numpy(), edge.long().numpy()

def fidelity(model,node, feature, edge, ground_edge, label):
    label = label[node]
    edge_weight = torch.ones(edge.shape[1], device=torch.device('cuda:0'))


    for i in range(edge.shape[1]):
        n1, n2 = edge[:,i]
        if (n1.item(), n2.item()) in ground_edge:
            edge_weight[i] = 0
    raw_predict = model(feature, edge, edge_weight = edge_weight)[node, :][label]
    predict = model(feature, edge)[node, :][label]
    print(predict)
    print(raw_predict)
    return  predict- raw_predict

def make_graph(dataset):
    edge_list = open(f"/home/steve305/kr/mmm/dataset/{dataset}/edge_list.txt", 'r')
    graph = nx.Graph()
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)

    return graph