import numpy as np
import torch
from collections import defaultdict
from arguments import args
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from utils import *
import tqdm
import pickle
from orbit_find import *
from torch_geometric.utils import k_hop_subgraph


# from orbit_find import *

class Explainer:
    def __init__(self, **kwargs):
        self.data_name = kwargs['data_name']
        self.data_path = f'../dataset/{kwargs["data_name"]}/'
        self.model = kwargs["model"]
        self.pred_bias = kwargs["pred"]
        self.answer_idx = kwargs['answer_idx']
        self.orbit_basis = None
        self.use_orbit = None
        self.similar_dict = None
        model_path = kwargs["model_path"]
        if not (os.path.exists(f'{model_path}basis/')):
            os.mkdir(f'{model_path}basis/')
        if not (os.path.exists(f'{model_path}score')):
            os.mkdir(f'{model_path}score')
        if not (os.path.exists(f'{model_path}result')):
            os.mkdir(f'{model_path}result')
        # model_linear_weight
        self.weight = self.model.ffn[1].weight
        self.basis_path = f'{model_path}basis/{args.o_lr}_{args.o_epochs}_{args.o_batch}.pkl'
        self.score_path = f'{model_path}score/{args.o_lr}_{args.o_epochs}_{args.o_batch}_{args.s_lr}_{args.s_epochs}.pkl'
        self.result_path = f'{model_path}result/{args.o_lr}_{args.o_epochs}_{args.o_batch}_{args.s_lr}_{args.s_epochs}.txt'

        self.load_orbit()



    def load_orbit(self):
        now_raw_orbit = np.genfromtxt(f'{self.data_path}orbit.txt', dtype=np.int32)
        args_sorts = np.argsort(now_raw_orbit[:, 0])
        raw_orbit = torch.LongTensor(now_raw_orbit[args_sorts][:, 1:] != 0)
        self.y_orbit = raw_orbit
        raw_orbit = raw_orbit[self.answer_idx]
        n_node = raw_orbit.shape[0]
        n_orbit = raw_orbit.shape[1]
        self.n_orbit = n_orbit
        use_orbit = [i for i in range(n_orbit)]
        sum_orbit = torch.sum(raw_orbit, axis=0)
        # for i in range(n_orbit):
        #     print(f'{i}: {sum_orbit[i]}')
        # exit()
        # for i in range(use_orbit):
        #     print(f'{i}: {sum_orbit[]}')
        for i in range(n_orbit):
            if sum_orbit[i] == raw_orbit.shape[0] or sum_orbit[i] == 0:
                use_orbit.remove(i)

        similar_dict = defaultdict(list)
        for i in range(n_orbit):
            for j in range(i + 1, n_orbit):
                similarity1 = torch.sum((raw_orbit[:, i] == raw_orbit[:, j]).float())/raw_orbit[:,i].shape[0]
      
                if similarity1 > 0.99:
                    similar_dict[i].append(j)
                if similarity1 > 0.95:
                    print(f'{i} {j}: {similarity1}')

    
        keys_list = list(similar_dict.keys())
        for i in keys_list:
            if i in use_orbit:
                for j in similar_dict[i]:
                    if j in use_orbit:
                        use_orbit.remove(j)
            else:
                del similar_dict[i]

        if args.num_model_layer < 4:
            use_orbit = [i for i in use_orbit if i != 24 and i != 42]

        if args.num_model_layer < 3:
            use_orbit = [i for i in use_orbit if
                         i not in [6, 8, 17, 18, 21, 26, 27, 30, 32, 34, 43, 45, 46, 47, 50, 53, 55, 56, 57, 59, 62,
                                   65]]
        self.similar_dict = similar_dict
        # The number of orbits
        self.use_orbit = use_orbit
        # Orbit existence
        self.raw_orbit_acc = dict()
        for i in range(n_orbit):
            self.raw_orbit_acc[i] = max((sum_orbit[i]/n_node).item(), ((n_node-sum_orbit[i])/n_node).item())

        print('As a result of prepositioning the orbit, we only use the following: ')
        print(' '.join(map(str, use_orbit)))
        print(f'Similar Orbit dict: {similar_dict}')
        print(f"total number of orbits: {len(use_orbit)}")

    def Dataset_and_Dataloader(self, x, y):
        # 0 대입시 배치 없이 학습이 진행
        class orbit_dataset(Dataset):
            def __init__(self):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                x = self.x[idx, :]
                y = self.y[idx, :]
                return x, y

        data_set = orbit_dataset()
        batch = args.o_batch
        if args.o_batch == 0:
            batch = len(data_set)
        return DataLoader(data_set, batch_size=batch, shuffle=True, drop_last=False)

    def orbit_basis_learning(self):
        # 추가할 여러가지 요소를 생각해보자.
        device = args.device
        repr = self.get_representation_vector()[self.answer_idx]
        y_orbit = self.y_orbit[self.answer_idx]
        use_orbit = self.use_orbit
        orbit_learner = torch.nn.Linear(repr.shape[1], self.n_orbit, bias=False).to(device)

        dataloader = self.Dataset_and_Dataloader(repr, y_orbit)
        #criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        criterion = FocalLoss()
        #optimizer = torch.optim.Adam(lr=args.o_lr, params=orbit_learner.parameters())
        optimizer = torch.optim.SGD(lr=args.o_lr, params=orbit_learner.parameters())
        global_loss = float('inf')
        for epoch in range(1, args.o_epochs + 1):
            progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}/{args.o_epochs}", unit="batch")
            batch = 0
            batch_loss = 0
            for x, y in progress_bar:
                batch += 1
                x = x.to(device)
                y = y.float()[:, self.use_orbit].to(device)

                z = orbit_learner(x)[:, self.use_orbit]

                optimizer.zero_grad()
                loss = criterion(z, y)
                loss.backward()

                batch_loss += loss.detach().item()
                #print(orbit_learner.weight[:10, ])

                optimizer.step()
            print(f'batch_loss: {batch_loss/batch}')
            if batch_loss/batch < global_loss:
                global_loss = batch_loss
                orbit_basis = orbit_learner.weight.to('cpu').detach()
        self.orbit_basis = torch.nn.functional.normalize(orbit_basis, p=2, dim=1)
        with torch.no_grad():
            repr = repr.to(device)
            z = orbit_learner(repr)
            z = (torch.sigmoid(z) >= 0.5).int()
            y_orbit = y_orbit.to(device)
            answer = torch.mean(torch.eq(z, y_orbit).float(), dim=0)
            print('-----Result of the orbit basis learning-----')
            unuse_orbit = []
            for o in use_orbit:
                print(f'ACC of the {o}_orbit: {answer[o]:.3f}')
                if answer[o] < self.raw_orbit_acc[o]:
                    unuse_orbit.append(o)
            for o in unuse_orbit:
                use_orbit.remove(o)
                #if answer[0] < len()


        self.use_orbit = use_orbit

        saving_dict = {'orbit_basis': self.orbit_basis, 'use_orbit': self.use_orbit, 'similar_dict': self.similar_dict}

        with open(self.basis_path, 'wb') as a:
            pickle.dump(saving_dict, a)

    def score_learning(self):
        print('-----Score learning start-----')
        # Basis load
        score_dict = dict()
        with open(self.basis_path, 'rb') as a:
            basis_dict = pickle.load(a)
        # orbit_basis = (orbit_number, hidden_d)
        self.orbit_basis, self.use_orbit = basis_dict['orbit_basis'], basis_dict['use_orbit']
        # weight = (num_class, hidden_d)
        n_class = self.weight.shape[0]
        for c in range(1, n_class):
            print(f"---{c}'th class---")
            score, choice_orbit = self.class_score_learing(c)
            score_dict[c] = [score, choice_orbit]
        with open(self.score_path, 'wb') as a:
            pickle.dump(score_dict, a)

    def class_score_learing(self, c):
        saving_dict = dict()
        device = args.device
        orbit_basis = self.orbit_basis
        use_orbit = self.use_orbit

        # orbit_basis = (use_orbit, hidden_d)

        class_weight = self.weight[c, :]
        sorted_orbit = torch.argsort(torch.squeeze(self.orbit_basis @ torch.unsqueeze(class_weight, 1)),descending=True)
        choice_orbit = []
        # for i in sorted_orbit:
        #     if i in use_orbit:
        #         print(f"{c}'th Class: {i} is selected!")
        #         choice_orbit = [i]
        #         break

        class_weight = torch.unsqueeze(class_weight, 1).to(device)

        # Dictionary to translate idx into orbit


        min_loss = float('inf')
        for n in range(1, args.n_concept ):
            pre_o, pre_loss = self.orbit_selection(choice_orbit, class_weight, orbit_basis)

            if pre_loss < min_loss:
            #if pre_loss != 100:
                min_loss = pre_loss
    
                print(f"{c}'th Class: {pre_o} is selected!")

                choice_orbit.append(pre_o)
            else:
                break
        choice_orbit.sort()
        orbit_basis = orbit_basis[choice_orbit, :].to(device)
        score_learner = PositiveLinear(orbit_basis.shape[0], 1).to(device)
        #optimizer = AdaBelief(lr=args.s_lr, params=score_learner.parameters(), print_change_log = False)
        optimizer = torch.optim.SGD(lr=args.s_lr, params=score_learner.parameters())
        now_loss = float('inf')
        criterion = nn.MSELoss(reduction='mean')
        for _ in range(3000):
            predict = score_learner(orbit_basis.T)
            loss = criterion(predict, class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < now_loss:
                score = torch.squeeze(score_learner.get_score()).to('cpu').detach().tolist()
                now_loss = loss.to('cpu').detach()

        #score = torch.squeeze(score_learner.get_score()).to('cpu').detach().tolist()
        if type(score) != list:
            score = [score]

        for i, j in zip(choice_orbit, score):
            print(f'Score of {i}: {j:.3f}')
        return score, choice_orbit

    def orbit_selection(self, choice_orbit, class_weight, orbit_basis):
        pre_o, pre_loss = 0, float('inf')
        device = args.device
        criterion = nn.MSELoss(reduction='mean')
        answer_predict = 0
        for idx in self.use_orbit:
            if idx in choice_orbit:
                continue
            with torch.no_grad():
                if orbit_basis[idx, :].to(device)@class_weight <0:
                    # print(f'idx: {idx}')
                    # print(orbit_basis[idx, :].to(device)@class_weight)
                    continue
            pre_basis = orbit_basis[choice_orbit + [idx], :].to(device)
            score_learner = PositiveLinear(pre_basis.shape[0], 1).to(device)
            optimizer = torch.optim.Adam(lr=args.s_lr, params=score_learner.parameters())

            #####
            # scores = torch.randn((len(choice_orbit) + 1, 1), requires_grad=True, device=device).float()
            # optimizer = torch.optim.RMSprop(lr=args.s_lr, params=[scores])
            ######

            now_loss = float('inf')
            for _ in range(args.s_epochs):
                # predict = pre_basis.T@torch.exp(scores)
                predict = score_learner(pre_basis.T)
                loss = criterion(predict, class_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if loss < now_loss:
                    now_loss = loss.to('cpu').detach()
            #print('************************************')
            if now_loss < pre_loss:
                pre_loss = now_loss
                pre_o = idx
                answer_predict = predict

        return pre_o, pre_loss

    def get_representation_vector(self):
        feature, label, edge = load_data(self.data_name)
        representation = self.model.get_emb(feature, edge)
        representation = representation.detach()

        return representation

    def get_class(self):
        feature, label, edge = load_data(self.data_name)
        return label

    def analyze(self, nodes):
        self.orbit_find_dict = [0, 1, find_2_orbit, find_3_orbit, find_4_orbit, find_5_orbit, find_6_orbit, find_7_orbit,
                               find_8_orbit, find_9_orbit, find_10_orbit,
                               find_11_orbit, find_12_orbit, find_13_orbit, find_14_orbit, find_15_orbit, find_16_orbit,
                               find_17_orbit, find_18_orbit, find_19_orbit, find_20_orbit, find_21_orbit, find_22_orbit,
                               find_23_orbit,
                               find_24_orbit, find_25_orbit, find_26_orbit, find_27_orbit, find_28_orbit, find_29_orbit,
                               find_30_orbit, find_31_orbit, find_32_orbit, find_33_orbit, find_34_orbit, find_35_orbit,
                               find_36_orbit, find_37_orbit, find_38_orbit, find_39_orbit, find_40_orbit,
                               find_41_orbit, find_42_orbit, find_43_orbit, find_44_orbit, find_45_orbit, find_46_orbit,
                               find_47_orbit,
                               find_48_orbit, find_49_orbit, find_50_orbit, find_51_orbit, find_52_orbit, find_53_orbit,
                               find_54_orbit, find_55_orbit, find_56_orbit, find_57_orbit, find_58_orbit, find_59_orbit,
                               find_60_orbit, find_61_orbit, find_62_orbit, find_63_orbit, find_64_orbit, find_65_orbit,
                               find_66_orbit, find_67_orbit, find_68_orbit, find_69_orbit, find_70_orbit, find_71_orbit,
                               find_72_orbit, find_73_orbit, find_74_orbit, find_75_orbit, find_76_orbit]

        self.find_gt_edge = {'bashapes': find_baco_gt, 'bac': find_baco_gt, 'tree_cycle': find_76_orbit }
        # model_path = self.model_path
        basis_path = self.basis_path
        score_path = self.score_path
        result_path = self.result_path
        model = self.model
        label = self.get_class()
        representation = self.get_representation_vector()
        feature, label, edge = load_data(self.data_name)


        with open(basis_path, 'rb') as a:
            basis_dict = pickle.load(a)

        with open(score_path, 'rb') as a:
            score_dict = pickle.load(a)
        txt = open(result_path, 'a')
        txt_list = []

        txt_list.extend([
                            f'----------------------------Explanation Result of basis learning(lr: {args.o_lr}, epoch: {args.o_epochs}, batch: {args.o_batch})'
                            f' and score learning(lr: {args.s_lr}, epoch: {args.s_epochs})----------------------------\n\n',
                            'Model-Level\n\n'])
        models = []
        for c in range(1, len(score_dict)+1):
            if c in args.except_class:
                continue
            score, idxs = score_dict[c]
            txt_list.append(f"{c}'th class\n")
            idxs = [(s, idx) for s, idx in zip(score, idxs)]
            idxs.sort(reverse=True)
            for i, j in enumerate(idxs):
                s, o = j
                txt_list.append(f'{i + 1} orbit: {o} ({s / sum(score):.3f})\n', )
                if i == 0:
                    models.append(o)
            txt_list.append('\n')
        txt.writelines(txt_list)
        orbit_basis, use_orbit, similar_dict = basis_dict['orbit_basis'], basis_dict['use_orbit'], basis_dict[
            'similar_dict']


        rep_basis = representation @ orbit_basis.T
        for c in range(1, len(score_dict)+1):
            score, orbits = score_dict[c]
     
            idx_score = rep_basis[:, orbits]*torch.FloatTensor(score)
  
            score_dict[c].append(idx_score)
 

        spars = []
        fids = []
        recalls = []
        sub_recalls = []
        edge_matrix = torch.zeros((feature.shape[0],feature.shape[0] ))
        model = model.to(args.device)
        graph = make_graph(self.data_name)

        
        feature = feature.to(args.device)
        edge = edge.to(args.device)
        for i in range(edge.shape[1]):
            n1, n2 = edge[:, i]
            edge_matrix[n1, n2] = 1
        for i in range(feature.shape[0]):
            edge_matrix[i, i] = 1
        edge_matrix = edge_matrix.to(args.device)

        label = label.to(args.device)
        for node in tqdm.tqdm(nodes):
            lab = label[node].item()
            if lab in args.except_class:
                continue
            print(f'------node: {node}-------')
            _, score, orbit_score = score_dict[lab]
            orbit_score = torch.argsort(orbit_score[node, :], descending=True)

            for o in orbit_score:
                pred_orbit = score[o]
                print(f'pred_orbit: {pred_orbit}')
                if pred_orbit in similar_dict.keys():
                    answer_orbit = [pred_orbit]+similar_dict[pred_orbit]
                else:
                    answer_orbit = [pred_orbit]
                fid, spar, recall, sub_recall = self.target_node(torch.tensor(node, device=args.device), answer_orbit, model, feature, edge, label, graph,edge_matrix )
    
            fids.append(fid)
            spars.append(spar)
            recalls.append(recall)
            sub_recalls.append(sub_recall)
            #fid = fidelity(model,400, feature, edge,[(400, 401),(400, 403),(400, 404), (401, 400), (401, 402), (401, 404)], label)
            #fiderity = fid(node, answer_orbit)
            print(f"{node}'th node's instance-level orbit: {pred_orbit}")
        txt.writelines(f"*****************************Instance-Level*****************************\n")
        txt.writelines(f"sparsity: {sum(spars)/len(spars)}\n")
        txt.writelines(f"fiderity: {sum(fids) / len(fids)}\n")
        txt.writelines(f"recall: {sum(recalls)/len(recalls)}\n")
        txt.writelines(f"sub_recall: {sum(sub_recalls) / len(sub_recalls)}\n")

        print(f"sparsity: {sum(spars)/len(spars)}, fiderity: {sum(fids) / len(fids)},recall: {sum(recalls)/len(recalls)},sub_recall: {sum(sub_recalls)/len(sub_recalls)}  ")
            # self.find_instance(node, lab, use_orbit, orbit_basis, similar_dict, score_dict, rep)

        return models
    # def find_instance(self, node, lab, use_orbit, orbit_basis, similar_dict, score_dict, rep):
    def target_node(self, node, answer_orbit, model, feature, edge, label, graph, edge_matrix):
        answer_fid = -float('inf')
        recall = 0
        sub_recall = 0
        fid = torch.Tensor(0)
        explain = []
        with torch.no_grad():
            for target_o in answer_orbit:
                if args.data_name == 'bashapes' and target_o == 10:
                    target_o = 58
                if args.data_name == 'bashapes' and  target_o == 28:
                    target_o = 56

                fid, can_explain = self.orbit_find_dict[target_o](graph, node, label, model, feature, edge, edge_matrix, choice_n = args.n_sample)
                for i in range(len(can_explain)):
                    n1, n2 = can_explain[i]
                    if torch.is_tensor(n1):
                        n1 = n1.detach().to('cpu').item()
                    elif torch.is_tensor(n2):
                        n2 = n2.detach().to('cpu').item()
                    can_explain[i] = (n1, n2)

                fid = torch.Tensor([fid])
 
                if answer_fid <= fid:
                    answer_fid = fid
                    explain = can_explain
        print(f'fid: {fid}')


        num_edge_subgraph = k_hop_subgraph(node.item(), args.num_model_layer, edge)[1].shape[1]
        num_explain_edge = len(explain)
        spar = 1-num_explain_edge/num_edge_subgraph


        if args.data_name in ['bashapes', 'bac']:

            gt_edge = self.find_gt_edge[args.data_name](graph, node, label)

        elif args.data_name == 'tree_cycle':
             _, gt_edge = self.find_gt_edge[args.data_name](graph, node, label, model, feature, edge, edge_matrix, choice_n =10)
        elif args.data_name == 'tree_cycle':
            gt_edge = find_cycle(graph, node)
        elif args.data_name == 'tree_grid':
            if label[node] == 1:
                gt_edge = label_1_grid(graph, node)
            elif label[node] == 2:
                gt_edge = label_2_grid(graph, node)
            elif label[node] == 3:
                gt_edge = label_3_grid(graph, node)

        recall = len(set(gt_edge)&set(explain))/len(gt_edge)

        if recall == 1:
            sub_recall = 1

        return fid.to('cpu').detach().item(), spar, recall, sub_recall








class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)

    def forward(self, input):
        # 양수 제약 조건 적용 (지수 함수를 사용하여 양수 값으로 변환)
        positive_weight = torch.exp(self.weight)

        #positive_weight = torch.nn.functional.relu(self.weight)
        return torch.matmul(input,  positive_weight)

    def get_score(self):
        return torch.exp(self.weight).detach()

