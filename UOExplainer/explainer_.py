import numpy as np
import torch
from collections import defaultdict
from arguments import args
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from utils import *
import tqdm
import pickle


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
        raw_orbit = np.genfromtxt(f'{self.data_path}orbit.txt', dtype=np.int32)
        args_sorts = np.argsort(raw_orbit[:, 0])
        raw_orbit = torch.LongTensor(raw_orbit[args_sorts][:, 1:] != 0)

        self.y_orbit = raw_orbit

        raw_orbit = raw_orbit[self.answer_idx]
        n_orbit = raw_orbit.shape[1]
        self.n_orbit = n_orbit
        use_orbit = [i for i in range(n_orbit)]
        sum_orbit = torch.sum(raw_orbit, axis=0)
        for i in range(n_orbit):
            if sum_orbit[i] == raw_orbit.shape[0] or sum_orbit[i] == 0:
                use_orbit.remove(i)

        similar_dict = defaultdict(list)
        for i in range(n_orbit):
            for j in range(i + 1, n_orbit):
                similarity1 = torch.nn.functional.cosine_similarity(raw_orbit[:, i].float().unsqueeze(0), raw_orbit[:, j].float().unsqueeze(0)).item()
                #print(f'{i} {j}:{similarity1}')
                if similarity1 > 0.99:
                    similar_dict[i].append(j)
                if similarity1 > 0.95:
                    print(f'{i} {j}: {similarity1}')

                #print(f'{i} {j}: {similarity[i][j]}')
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
        #criterion = torch.nn.BCEWithLogitsLoss()
        criterion = FocalLoss()
        optimizer = torch.optim.Adam(lr=args.o_lr, params=orbit_learner.parameters())
        for epoch in range(1, args.o_epochs + 1):
            progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}/{args.o_epochs}", unit="batch")
            for x, y in progress_bar:
                x = x.to(device)
                y = y.float()[:, self.use_orbit].to(device)
                z = orbit_learner(x)[:, self.use_orbit]
                optimizer.zero_grad()
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            repr = repr.to(device)
            z = orbit_learner(repr)
            z = (torch.sigmoid(z) >= 0.5).int()
            y_orbit = y_orbit.to(device)
            answer = torch.mean(torch.eq(z, y_orbit).float(), dim=0)
            print('-----Result of the orbit basis learning-----')
            for o in use_orbit:
                print(f'ACC of the {o}_orbit: {answer[o]:.3f}')


        orbit_basis = orbit_learner.weight.to('cpu').detach()
        self.orbit_basis = torch.nn.functional.normalize(orbit_basis, p=2, dim=1)

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
        class_weight = torch.unsqueeze(class_weight, 1).to(device)

        # Dictionary to translate idx into orbit
        choice_orbit = []
        min_loss = float('inf')
        for n in range(1, args.n_concept + 1):
            pre_o, pre_loss = self.orbit_selection(choice_orbit, class_weight, orbit_basis)
            print(pre_loss)
            if pre_loss < min_loss:
            #if pre_loss != 100:
                min_loss = pre_loss
                print(pre_loss)
                print(f"{c}'th Class: {pre_o} is selected!")

                choice_orbit.append(pre_o)
            else:
                break
        choice_orbit.sort()
        orbit_basis = orbit_basis[choice_orbit, :].to(device)
        score_learner = PositiveLinear(orbit_basis.shape[0], 1).to(device)
        #optimizer = AdaBelief(lr=args.s_lr, params=score_learner.parameters(), print_change_log = False)
        optimizer = torch.optim.Adam(lr=args.s_lr, params=score_learner.parameters())
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
        print('*****************************')
        print(torch.squeeze(predict))
        print(torch.squeeze(class_weight))
        print('*****************************')
        print(choice_orbit, score)
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
                #print(loss)

                if loss < now_loss:
                    now_loss = loss.to('cpu').detach()
            #print('************************************')
            if now_loss < pre_loss:
                pre_loss = now_loss
                pre_o = idx
                answer_predict = predict
            #print(now_loss)
        #print(torch.squeeze(answer_predict))
        #print(torch.squeeze(class_weight))
        #print('----------------------')
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
        # model_path = self.model_path
        basis_path = self.basis_path
        score_path = self.score_path
        result_path = self.result_path
        model = self.model
        label = self.get_class()
        representation = self.get_representation_vector()

        with open(basis_path, 'rb') as a:
            basis_dict = pickle.load(a)

        with open(score_path, 'rb') as a:
            score_dict = pickle.load(a)
        txt = open(result_path, 'w')
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

        # rep_basis = (num_node, num_orbit)
        rep_basis = representation @ orbit_basis.T
        for c in range(1, len(score_dict)+1):
            score, orbits = score_dict[c]
            idx_score = rep_basis[:, orbits]*torch.FloatTensor(score)
            print(score)
            score_dict[c].append(idx_score)
            print(torch.FloatTensor(score)@orbit_basis[orbits, :])
            print(self.weight[c, :])
            print('-------------------')

        spar = []
        fid = []
        rec = []

        for node in tqdm.tqdm(nodes):
            lab = label[node].item()
            if lab in args.except_class:
                continue

            _, score, orbit_score = score_dict[lab]

            orbit_score = torch.argsort(orbit_score[node, :], descending=True)
            pred_orbit = score[orbit_score[0]]

            print(f"{node}'th node's instance-level orbit: {pred_orbit}")

            self.find_instance(node, lab, use_orbiƒt, orbit_basis, similar_dict, score_dict, rep)
        return models
    # def find_instance(self, node, lab, use_orbit, orbit_basis, similar_dict, score_dict, rep):

    def score_learnings(self):
        print('-----Score learning start-----')
        # Basis load
        score_dict = dict()
        with open(self.basis_path, 'rb') as a:
            basis_dict = pickle.load(a)
        # orbit_basis = (orbit_number, hidden_d)
        self.orbit_basis, self.use_orbit = basis_dict['orbit_basis'], basis_dict['use_orbit']
        pred_bias = self.pred_bias[self.answer_idx, :]
        rep = self.get_representation_vector()[self.answer_idx, :]
        # weight = (num_class, hidden_d)
        n_class = self.weight.shape[0]
        print(n_class)
        exit()
        for c in range(1, n_class):
            print(f"---{c}'th class---")
            score, choice_orbit = self.class_score_learning_(c, rep, pred_bias)
            score_dict[c] = [score, choice_orbit]
        with open(self.score_path, 'wb') as a:
            pickle.dump(score_dict, a)

    def class_score_learning_(self, c, rep, pred_bias):
        saving_dict = dict()
        device = args.device
        orbit_basis = self.orbit_basis
        use_orbit = self.use_orbit
        pred_bias = pred_bias[:,c]

        # orbit_basis = (use_orbit, hidden_d)

        class_weight = self.weight[c, :]
        class_weight = torch.unsqueeze(class_weight, 1).to(device)

        # Dictionary to translate idx into orbit
        choice_orbit = []
        min_loss = float('inf')
        for n in range(1, args.n_concept + 1):
            pre_o, pre_loss = self.orbit_selection_(choice_orbit, class_weight, orbit_basis, rep, pred_bias)
            if pre_loss < min_loss:
                print(f"{c}'th Class: {pre_o} is selected!")
                choice_orbit.append(pre_o)
            else:
                break
        choice_orbit.sort()
        orbit_basis = orbit_basis[choice_orbit, :].to(device)
        score_learner = PositiveLinear(orbit_basis.shape[0], 1).to(device)
        optimizer = torch.optim.Adam(lr=args.s_lr, params=score_learner.parameters())
        now_loss = float('inf')
        criterion = nn.MSELoss(reduction='mean')

        for _ in range(1000):
            predict = score_learner(orbit_basis.T)
            loss = criterion(predict, class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < now_loss:
                now_loss = loss.to('cpu').detach()
        score = torch.squeeze(score_learner.get_score().to('cpu').detach()).tolist()
        for i, j in zip(choice_orbit, score):
            print(f'Score of {i}: {j:.3f}')
        return score, choice_orbit

    def orbit_selection_(self, choice_orbit, class_weight, orbit_basis, rep, pred_bias):
        pre_o, pre_loss = 0, float('inf')
        device = args.device
        criterion = nn.MSELoss(reduction='sum')
        repr = self.get_representation_vector()[self.answer_idx]
        rep = rep.to(device)
        pred_bias = pred_bias.to(device)

        for idx in self.use_orbit:
            if idx in choice_orbit:
                continue

            pre_basis = orbit_basis[choice_orbit + [idx], :].to(device)
            score_learner = PositiveLinear(pre_basis.shape[0], 1).to(device)
            optimizer = torch.optim.Adam(lr=args.s_lr, params=score_learner.parameters())
            now_loss = float('inf')
            for _ in range(args.s_epochs):
                predict = score_learner(pre_basis.T)
                ret = torch.squeeze(torch.matmul(rep, predict))
                loss = criterion(ret, pred_bias)
                #print(ret, pred_bias)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss < now_loss:
                    now_loss = loss.to('cpu').detach()

            if now_loss < pre_loss:
                pre_loss = now_loss
                pre_o = idx
        return pre_o, pre_loss


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
        return input @ positive_weight

    def get_score(self):
        return torch.exp(self.weight)
