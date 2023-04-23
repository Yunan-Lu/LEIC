import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, Categorical, Dirichlet
from torch.distributions.kl import kl_divergence
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph

def get_net_params(*args):
    res = []
    for x in args:
        res.extend(list(x.parameters()))
    return res

class PosNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.sp = nn.Softplus()

    def forward(self, x):
        return self.sp(self.fc(x)) + 1e-4

class GConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1))**(0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, rowLap, colLap=None):
        support = torch.mm(input, self.weight)
        output = rowLap @ support
        if colLap is not None:
            output = output @ colLap

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GaussianMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mean_net = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.std_net = PosNet(in_dim, out_dim)
    def forward(self, x):
        return self.mean_net(x), self.std_net(x)

class GaussianGConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mean_net = GConv(in_dim, out_dim)
        self.std_net = GConv(in_dim, out_dim)
        self.sp = nn.Softplus()
    def forward(self, x, adj):
        return self.mean_net(x, adj), self.sp(self.std_net(x, adj)) + 1e-4

def compute_Laplacian(data, n_neighbors):
    '''data: (n_samples, n_features)'''
    adj = kneighbors_graph(data, n_neighbors, include_self=True, metric='cosine').toarray()
    adj = adj + adj.T
    adj[adj != 0] = 1
    Lap = adj.sum(1, keepdims=True)**(-.5) * adj * adj.sum(0, keepdims=True)**(-.5)
    return adj, Lap


class LEIC(BaseEstimator):
    '''
    Parameters::
        cdim: int, default=None (i.e., n_labels+1)
            Number of Gaussian components.
        zdim: int, default=64
            Dimension of the joint implicit representation.
        lam: float, default=4
            Strength of simple label information, q(d) = Dir(lam*y + ...).
        pyd: torch.distributions.Distribution, default=Bernoulli
            Probability distribution for generating simple labels from label distributions, 
            which by default is Bernoulli, i.e., the case where the simple labels are logical.
        trace_step: int, default=20
            Result recording step, which facilitates early-stop technique.
        verbose: int, default=0 
            How many intermediate ELBO values will be printed during training.
        lr: float, default=1e-3
            Learning rate of Adam.
        max_iter: int, default=500
            Maximum iterations of Adam.
    --------------------------------------------------------------
    Attributes::
        label_distribution_: ndarray of shape (n_samples, n_labels)
            Recovered label distributions.
        trace_: dict of label_distribution_
            Recovered label distributions on different epoch.
    --------------------------------------------------------------
    Methods::
        fit(X, Y): training the model with feature X and simple label Y.
    --------------------------------------------------------------
    Examples::
        >>> Drec = LEIC(trace_step=np.inf).fit(X, Y).label_distribution_
        >>> evaluate(Drec, ground_truth)
        >>> Drec_trace = LEIC(trace_step=20).fit(X, Y).trace_
        >>> for k in Drec_trace.keys():
        >>>     evaluate(Drec_trace[k], ground_truth)
    '''

    def __init__(self, cdim=None, zdim=64, lam=4, pyd=Bernoulli, lr=1e-3,
        max_iter=500, trace_step=20, verbose=0, random_state=123):
        assert verbose < max_iter
        self.cdim = cdim
        self.zdim = zdim
        self.lam = lam
        self.pyd = pyd
        self.lr = lr
        self.max_iter = max_iter
        self.trace_step = trace_step
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, Y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        self.cdim = Y.shape[1] + 1 if self.cdim is None else self.cdim
        yadj, yLap = compute_Laplacian(Y, 20)
        X = MinMaxScaler().fit_transform(X)
        X, Y, yLap, yadj = (torch.FloatTensor(x) for x in [X, Y, yLap, yadj])
        (n, xdim), ydim, cdim, zdim = X.shape, Y.shape[1], self.cdim, self.zdim
        
        self.pd2zNet = GaussianMLP(ydim, zdim * cdim)
        self.pz2xNet = GaussianMLP(zdim, xdim)
        self.qx2zNet = GaussianGConv(xdim, zdim)
        self.qx2dNet = PosNet(xdim, ydim)
        self.qx2cNet = GConv(xdim, cdim)

        def elbo():
            pc = Categorical(torch.ones(n, cdim))
            qz = Normal(*self.qx2zNet(X, yLap)) # (n, zdim)
            zsam = qz.rsample() # (n, zdim)
            pd = Dirichlet(torch.ones(n, ydim))
            
            qc_param = torch.softmax(self.qx2cNet(X, yLap), dim=1) # (n, cdim)
            qc = Categorical(qc_param)
            ALc = (qc_param.detach().view(n, cdim, 1) * Y.view(n, 1, ydim)).permute(1,2,0) @ Y # (cdim, ydim, ydim)
            nAL = (qc_param.detach().view(n, cdim, 1, 1) * ALc).sum(1) # (n, ydim, ydim)
            rowsum = nAL.sum(1, keepdims=True) ** (-0.5) # (n, 1, ydim)
            nAL = rowsum.permute(0,2,1) * nAL * rowsum # (n, ydim, ydim)
            _Dconc = self.qx2dNet(X)
            Dconc = (_Dconc.unsqueeze(1) @ nAL).squeeze(1) # (n, ydim)

            qd = Dirichlet(Dconc + self.lam * Y)
            dsam = qd.rsample() # (n, ydim)
            
            _Zmeans, _Zstds = self.pd2zNet(dsam)
            Zmeans = _Zmeans.view(n, cdim, zdim).transpose(0,1)
            Zstds = _Zstds.view(n, cdim, zdim).transpose(0,1)
            pz = Normal(Zmeans, Zstds) # (cdim, n, zdim)
            
            recY_quality = self.pyd(dsam).log_prob(Y).sum()
            recA_quality = Bernoulli(torch.sigmoid(zsam@zsam.T)).log_prob(yadj).sum()
            recX_quality = Normal(*self.pz2xNet(zsam)).log_prob(X).sum()
            rec_quality = recY_quality + recA_quality + recX_quality
            klc_mean = kl_divergence(qc, pc).sum()
            klz = kl_divergence(qz, pz) # (cdim, n, zdim)
            klz_mean = (klz * qc_param.T.unsqueeze(-1)).sum()
            kld_mean = kl_divergence(qd, pd).sum()
            return rec_quality - (klc_mean + klz_mean + kld_mean), qd.mean.detach().numpy()
        
        params = get_net_params(self.pd2zNet, self.pz2xNet, 
                    self.qx2zNet, self.qx2cNet, self.qx2dNet)
        optimizer = torch.optim.Adam(params, lr=self.lr)
        self.trace_ = dict()
        for epoch in range(self.max_iter+1):
            optimizer.zero_grad()
            elbo_value, self.label_distribution_ = elbo()
            loss = -elbo_value
            loss.backward()
            optimizer.step()
            if (not np.isinf(self.trace_step)) and (epoch % self.trace_step == 0):
                self.trace_[epoch] = self.label_distribution_
            if (self.verbose > 0) and (epoch % (self.max_iter // self.verbose) == 0):
                with torch.no_grad():
                    print("* epoch: %4d, elbo: %.3f" % (epoch, elbo_value.item()))
        return self