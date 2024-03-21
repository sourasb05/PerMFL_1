from torch.optim import Optimizer


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    # Step : single optimization step (parameter update)
    # Closure : A closure that reevaluates the model and returns the loss. Optional to most optimizers.

    def step(self, closure=None, beta=0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if (beta != 0):
                    p.data.add_(-beta, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)
        return loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss

class L2GDopt(Optimizer):
    def __init__(self, params, lr=0.01, p_0=0, p_j=0):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, p_0=p_0, p_j=p_j)
        super(L2GDopt, self).__init__(params, defaults)

    def step(self, local_weight_updated, lr, p_0, p_j, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        
        for theta in local_weight_updated:
            theta.data = theta.data - (lr / ((1-p_0)*(1-p_j))) * theta.grad.data
        return local_weight_updated, loss

    

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, alpha=0.01, lamda=0.1):
        # self.local_weight_updated = local_weight # w_i,K
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if lamda < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))

        defaults = dict(alpha=alpha, lamda=lamda)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for theta, localweight in zip(group['params'], weight_update):
                theta.data = theta.data - group['alpha'] * ( theta.grad.data + group['lamda'] * (theta.data - localweight.data))
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for theta, localweight in zip(group['params'], weight_update):
                theta.data = localweight.data
        # return  p.data
        return group['params']


class pFedMe_original_Optimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMe_original_Optimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = p.data - group['lr'] * (
                        p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, beta=1, n_k=1):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss


class Ditto_Optimizer(Optimizer):
    def __init__(self, params, eta=0.01, ditto_lambda=0.1):
        self.eta = eta
        self.ditto_lambda = ditto_lambda
        # self.local_weight_updated = local_weight # w_i,K
        if eta < 0.0:
            raise ValueError("Invalid learning rate: {}".format(eta))
        if ditto_lambda < 0.0:
            raise ValueError("Invalid learning rate: {}".format(eta))

        defaults = dict(eta=eta, ditto_lambda=ditto_lambda)
        super(Ditto_Optimizer, self).__init__(params, defaults)

    def step(self, v_k, w, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for l_param, g_param in zip(v_k, w):
            l_param.data = l_param.data - self.eta * ( l_param.grad.data 
                            + self.ditto_lambda *(l_param.data - g_param.data))
        return v_k, loss

    

