import torch
from torch.distributions import Normal


class CriticNN(torch.nn.Module):
    def __init__(self, in_size, h_size, out_size, n_h_layers, init_func=None):
        super().__init__()
        self.in_size = in_size

        h_layers = [
            torch.nn.Linear(in_size, h_size),
            torch.nn.ReLU(),
        ]
        for _ in range(max(0, n_h_layers - 2)):
            h_layers.extend([
                torch.nn.Linear(h_size, h_size),
                torch.nn.ReLU(),
            ])

        self.sequential = torch.nn.Sequential(
            *h_layers,
            torch.nn.Linear(h_size, out_size),
        )

        def init_func_default(layer):
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight,
                                              gain=0.5)
                torch.nn.init.zeros_(layer.bias)
        init_func = init_func or init_func_default
        self.apply(init_func)

    def __call__(self, x: torch.Tensor):
        return self.sequential(x)

    def predict(self, x: torch.Tensor):
        return self(x)


class ActorNN(torch.nn.Module):
    def __init__(self, in_size, h_size, out_size, n_h_layers, init_func=None):
        super().__init__()
        self.in_size = in_size

        h_layers = [
            torch.nn.Linear(in_size, h_size),
            torch.nn.ReLU(),
        ]
        for _ in range(max(0, n_h_layers - 2)):
            h_layers.extend([
                torch.nn.Linear(h_size, h_size),
                torch.nn.ReLU(),
            ])

        self.sequential = torch.nn.Sequential(
            *h_layers,
        )
        self.mean_out = torch.nn.Linear(h_size, out_size)
        self.var_out = torch.nn.Linear(h_size, out_size)
        self.softplus = torch.nn.Softplus()

        def init_func_default(layer):
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.5)
                torch.nn.init.zeros_(layer.bias)
        init_func = init_func or init_func_default
        self.apply(init_func)

    def __call__(self, x: torch.Tensor):
        out = self.sequential(x)
        return self.mean_out(out), self.softplus(self.var_out(out))

    def predict(self, x: torch.Tensor):
        mean, var = self(x)
        dist = Normal(mean, var)
        action = dist.sample()
        probab = dist.log_prob(action)
        return action, probab, var

