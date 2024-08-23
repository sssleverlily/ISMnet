import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ISMnet(torch.nn.Module):
    def __init__(self, data, args):
        super(ISMnet, self).__init__()
        self.maxOrder = args.maxOrder
        self.hubOrder = args.hubOrder
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()
        if self.maxOrder >= self.hubOrder:
            self.order_list = list(range(0, self.hubOrder)) + list(range(self.hubOrder + 1, self.maxOrder + 1))
        else:
            self.order_list = list(range(0, self.maxOrder + 1))
        for i in range(len(self.order_list)):
            self.lin_in.append(Linear(data.num_features, args.hidden))
            self.hgc.append(cheby_prop(args.K, args.alpha))
            # self.hgc.append(Jacobi_prop(args.K, args.alpha, args.a, args.b))

        self.lin_out1 = Linear(args.hidden * len(self.order_list), 16)  # self.maxOrder
        self.lin_out2 = Linear(16, 1)

        # self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.maxOrder):
            self.lin_in[i].reset_parameters()
        self.lin_out1.reset_parameters()
        self.lin_out2.reset_parameters()

    def forward(self, data):

        x, HL = data.x, data.HL
        x_concat = torch.tensor([]).to(device)

        for id_, order in enumerate(self.order_list):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[id_](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[id_](xx, HL[order])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        x_concat = F.leaky_relu(x_concat)
        x_concat = self.lin_out1(x_concat)

        x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
        # x_concat = F.relu(x_concat)
        x_concat = self.lin_out2(x_concat)

        return F.leaky_relu(x_concat)


class cheby_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, bias=True, **kwargs):
        super(cheby_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        # filterWeights = initFilterWeight(Init, alpha, K, Gamma)
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.exp_a = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

        torch.nn.init.ones_(self.exp_a)

    def forward(self, x, HL):
        EXP_A = F.relu(self.exp_a)
        FW = (self.fW)

        # hidden = x * (self.fW[0])
        # for k in range(self.K):
        #     x = matmul(HL, x, reduce=self.aggr)
        #     gamma = self.fW[k + 1]
        #     hidden = hidden + gamma * x

        temp = [x, matmul(HL, x)]
        for k in range(2, self.K + 1):
            temp.append(2 * matmul(HL, temp[k - 1]) - temp[k - 2])

        # hidden=torch.zeros(x.size(), device=device)
        hidden = FW[0] * temp[0]
        for k in range(1, self.K + 1):
            hidden = hidden + FW[k] / (k ** EXP_A) * temp[k]

        # hidden = torch.stack(temp, dim=0)
        # hidden = torch.einsum('bij,jk->ik', hidden,t )
        self.beta = 0
        for k in range(1, self.K, 4):
            self.beta = self.beta + k * FW[k] / (k ** EXP_A)
        for k in range(3, self.K, 4):
            self.beta = self.beta - k * FW[k] / (k ** EXP_A)

        return hidden

    def __repr__(self):
        return '{}(K={}, exp_a={}, beta={}, filterWeights={})'.format(self.__class__.__name__, self.K, self.exp_a,
                                                                      self.beta, self.fW)


class Jacobi_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, a=1.0, b=1.0, **kwargs):
        super(Jacobi_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.a = a
        self.b = b
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.exp_a = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

        torch.nn.init.ones_(self.exp_a)

    def forward(self, x, HL):
        EXP_A = F.relu(self.exp_a)
        FW = F.relu(self.fW)

        temp = [x, 0.5 * (self.a - self.b) * x + 0.5 * (self.a + self.b + 2) * matmul(HL, x)]
        for k in range(2, self.K + 1):
            theta_1 = (2 * k + self.a + self.b) * (2 * k + self.a + self.b - 1) / (2 * k * (k + self.a + self.b))
            theta_2 = (2 * k + self.a + self.b - 1) * (self.a ** 2 - self.b ** 2) / (
                    2 * k * (k + self.a + self.b) * (2 * k + self.a + self.b - 2))
            theta_3 = (k + self.a - 1) * (k + self.b - 1) * (2 * k + self.a + self.b) / (
                    k * (k + self.a + self.b) * (2 * k + self.a + self.b - 2))
            temp.append(theta_1 * matmul(HL, temp[k - 1]) + theta_2 * temp[k - 1] - theta_3 * temp[k - 2])

        hidden = FW[0] * temp[0]
        for k in range(1, self.K + 1):
            hidden = hidden + FW[k] / (k ** EXP_A) * temp[k]

        return hidden

    def __repr__(self):
        return '{}(K={}, exp_a={}, filterWeights={})'.format(self.__class__.__name__, self.K, self.exp_a, self.fW)


class CNN1(nn.Module):
    def __init__(self, data, args):
        super(CNN1, self).__init__()
        self.conv1 = nn.Linear(in_features=data.num_features, out_features=args.hidden)
        self.conv2 = nn.Linear(in_features=args.hidden, out_features=32)
        # self.MaxPool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        x = data.x
        x = x.float()
        x = F.relu(self.conv1(x))
        # x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        # x = self.MaxPool(x)
        # x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
