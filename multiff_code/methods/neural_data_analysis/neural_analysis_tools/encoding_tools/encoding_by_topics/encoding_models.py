import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold


# =========================
# Core RNN module
# =========================
class _PoissonGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        h, _ = self.rnn(x)
        rate = torch.exp(self.readout(h))  # ensure positive rate
        return rate


# =========================
# RNN Model class
# =========================
class RNNModel:
    """
    Sequence model for neural encoding using GRU + Poisson likelihood.

    Assumes task provides:
        - binned_feats (DataFrame)
        - binned_spikes (DataFrame)
        - get_cv_groups_for_design(...)
    """

    def __init__(
        self,
        hidden_dim=64,
        n_epochs=20,
        lr=1e-3,
        device='cpu',
        verbose=True,
    ):
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.verbose = verbose

    # =========================
    # Utils
    # =========================
    @staticmethod
    def _poisson_nll(rate, y, eps=1e-8):
        return torch.mean(rate - y * torch.log(rate + eps))

    def _build_sequences(self, task, unit_idx):
        """
        Convert flat time series → list of sequences using grouping.
        """
        X = task.binned_feats.to_numpy()
        y = task.binned_spikes.iloc[:, unit_idx].to_numpy()

        groups = task.get_cv_groups_for_design(task.binned_feats)
        if groups is None:
            raise ValueError('RNN requires grouping (e.g. new_segment)')

        X_list, y_list = [], []

        for g in np.unique(groups):
            idx = (groups == g)
            X_list.append(X[idx])
            y_list.append(y[idx][:, None])

        return X_list, y_list, groups

    def _to_tensor_list(self, X_list, y_list):
        X_t = [torch.tensor(x, dtype=torch.float32, device=self.device) for x in X_list]
        y_t = [torch.tensor(y, dtype=torch.float32, device=self.device) for y in y_list]
        return X_t, y_t

    def _train_model(self, model, X_list, y_list):
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0

            for x, y in zip(X_list, y_list):
                x = x.unsqueeze(0)  # (1, T, D)
                y = y.unsqueeze(0)  # (1, T, 1)

                rate = model(x)
                loss = self._poisson_nll(rate, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            if self.verbose:
                print(f'[RNN] epoch {epoch} loss: {total_loss:.4f}')

    @torch.no_grad()
    def _evaluate(self, model, X_list, y_list):
        model.eval()

        y_all = torch.cat(y_list, dim=0)
        mean_rate = torch.mean(y_all)

        ll_model, ll_null = 0.0, 0.0

        for x, y in zip(X_list, y_list):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            rate = model(x)

            ll_model += torch.sum(y * torch.log(rate + 1e-8) - rate).item()
            ll_null += torch.sum(y * torch.log(mean_rate + 1e-8) - mean_rate).item()

        pseudo_r2 = 1 - (ll_model / ll_null)
        return pseudo_r2

    # =========================
    # Public API
    # =========================
    def fit(self, task, unit_idx):
        """
        Train on all data (no CV).
        """
        task.collect_data(exists_ok=True)

        X_list, y_list, _ = self._build_sequences(task, unit_idx)
        X_list, y_list = self._to_tensor_list(X_list, y_list)

        input_dim = X_list[0].shape[1]
        model = _PoissonGRU(input_dim, self.hidden_dim).to(self.device)

        self._train_model(model, X_list, y_list)

        return model

    def crossval(self, task, unit_idx, n_folds=5):
        """
        GroupKFold cross-validation.
        """
        task.collect_data(exists_ok=True)

        X = task.binned_feats.to_numpy()
        y = task.binned_spikes.iloc[:, unit_idx].to_numpy()
        groups = task.get_cv_groups_for_design(task.binned_feats)

        if groups is None:
            raise ValueError('RNN crossval requires grouping')

        gkf = GroupKFold(n_splits=n_folds)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            if self.verbose:
                print(f'[RNN] Fold {fold}')

            def make_seq(idx_set):
                X_list, y_list = [], []
                for g in np.unique(groups[idx_set]):
                    idx = (groups == g)
                    X_list.append(X[idx])
                    y_list.append(y[idx][:, None])
                return X_list, y_list

            X_train, y_train = make_seq(train_idx)
            X_test, y_test = make_seq(test_idx)

            X_train, y_train = self._to_tensor_list(X_train, y_train)
            X_test, y_test = self._to_tensor_list(X_test, y_test)

            input_dim = X_train[0].shape[1]
            model = _PoissonGRU(input_dim, self.hidden_dim).to(self.device)

            self._train_model(model, X_train, y_train)

            r2 = self._evaluate(model, X_test, y_test)

            if self.verbose:
                print(f'[RNN] Fold {fold} pseudo-R2: {r2:.4f}')

            scores.append(r2)

        return {
            'mean_pseudo_r2': float(np.mean(scores)),
            'all_folds': scores,
        }