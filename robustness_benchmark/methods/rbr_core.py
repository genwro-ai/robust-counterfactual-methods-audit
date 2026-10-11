import numpy as np
import torch
from sklearn.utils import check_random_state

from robustness_benchmark.methods.feature_constraints import search_projection
from robustness_benchmark.methods.rbr_loss import RBRLoss


class RBR:
    """Class for generate counterfactual samples for framework: Wachter"""

    DECISION_THRESHOLD = 0.5

    def __init__(
        self,
        model,
        train_data,
        y_target=1,
        num_cfacts=10,
        max_iter=1000,
        random_state=None,
        device="cuda",
        frozen=None,
        schema=None,
    ):
        self.random_state = check_random_state(random_state)
        self.frozen = list(frozen or [])
        self.schema = schema
        self.project = None
        self.model = model
        self.max_iter = max_iter
        self.y_target = y_target
        if "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = "cpu"
        self.feasible = True
        self.train_data = torch.tensor(train_data).float()
        self.train_label = self.make_prediction(self.train_data).to(self.device)
        self.train_data = self.train_data.to(self.device)
        self.num_cfacts = num_cfacts

    def make_prediction(self, x):
        return torch.tensor(self.model.predict(x.cpu().detach().numpy()))

    def dist(self, a, b):
        return torch.linalg.norm(a - b, ord=1, axis=-1)

    def projected(self, x):
        """``x`` as the registry will score it: projected on constrained schemas."""

        if self.project is None:
            return x
        return torch.as_tensor(self.project(x.detach().cpu().numpy()), dtype=x.dtype)

    def projected_if_valid(self, x, fallback=None):
        """The projected ``x`` if it is classified as the target, else ``fallback``."""

        point = self.projected(x).clone()
        return point if self.make_prediction(point) == self.y_target else fallback

    def find_x_boundary(self, x, k=None):
        k = k or self.num_cfacts
        x_label = self.make_prediction(x)

        d = self.dist(self.train_data, x)
        order = torch.argsort(d)
        x_cfact_list = self.train_data[order[self.train_label[order] == (1 - x_label)]][
            :k
        ]
        if self.frozen:
            x_cfact_list = x_cfact_list.clone()
            x_cfact_list[:, self.frozen] = x[self.frozen]
        best_x_b = None
        best_dist = torch.tensor(float("inf"))

        for x_cfact in x_cfact_list:
            lambd_list = torch.linspace(0, 1, 100)[:, None]
            line = self.projected((1 - lambd_list) * x + lambd_list * x_cfact)
            for x_b in line:
                label = self.make_prediction(x_b)
                if label == 1 - x_label:
                    dist = self.dist(x, x_b)
                    if dist < best_dist:
                        best_x_b = x_b
                        best_dist = dist
                    break
        return best_x_b, best_dist

    def uniform_ball(self, x, r, n, random_state=None):
        # muller method
        rng = check_random_state(random_state)
        d = len(x)
        V_x = rng.randn(n, d)
        V_x = V_x / np.linalg.norm(V_x, axis=1).reshape(-1, 1)
        V_x = V_x * (rng.random(n) ** (1.0 / d)).reshape(-1, 1)
        V_x = V_x * r + x.numpy()
        return torch.from_numpy(V_x).to(self.device)

    def feasible_set(self, x, radius=0.3, num_samples=20, random_state=None):
        return self.uniform_ball(x.cpu(), radius, num_samples, random_state)

    def simplex_projection(self, x, delta):
        """
        Euclidean projection on a positive simplex
        """
        (p,) = x.shape
        if torch.linalg.norm(x, ord=1) == delta and torch.all(x >= 0):
            return x
        u, _ = torch.sort(x, descending=True)
        cssv = torch.cumsum(u, 0)
        rho = torch.nonzero(
            u * torch.arange(1, p + 1).to(self.device) > (cssv - delta)
        )[-1, 0]
        theta = (cssv[rho] - delta) / (rho + 1.0)
        w = torch.clip(x - theta, min=0)
        return w

    def projection(self, x, delta):
        """
        Euclidean projection on an L1-ball
        """
        x_abs = torch.abs(x)
        if x_abs.sum() <= delta:
            return x

        proj = self.simplex_projection(x_abs, delta=delta)
        proj *= torch.sign(x)

        return proj

    def optimize(self, x, delta, loss_fn, theta=0.7, beta=1, verbose=False):
        x_t = x.detach().clone()
        x_t.requires_grad_(True)

        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10
        last_valid = self.projected_if_valid(x_t) if self.project is not None else None

        for t in range(self.max_iter):
            if x_t.grad is not None:
                x_t.grad.data.zero_()
            F, pe, op = loss_fn(x_t)

            F.backward()
            if self.frozen:
                x_t.grad[self.frozen] = 0.0
            if verbose:
                print(f"Iteration {t + 1}/{self.max_iter}")
                print(x_t)
                print(x_t.grad)
                print(F)
                print(op)
                print(pe)
                print(self.dist(x_t.detach().cpu(), self.x0.detach().cpu()))

            if torch.ge(self.dist(x_t.detach(), self.x0), delta):
                break

            with torch.no_grad():
                x_new = (
                    x_t
                    - 1 / torch.sqrt(torch.tensor(1e3, device=self.device)) * x_t.grad
                )
                x_new = (
                    self.projection(x_new - self.x0, delta) + self.x0
                )  # shift to origin before project

            for i, elem in enumerate(x_new.data):
                x_t.data[i] = elem
            if self.project is not None:
                last_valid = self.projected_if_valid(x_t, last_valid)

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum

            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0

            min_loss = min(min_loss, loss_sum)

        if last_valid is not None:
            return F, last_valid
        return F, x_t.detach()

    def fit_instance(
        self,
        x0,
        num_samples,
        perturb_radius,
        delta_plus,
        sigma,
        epsilon_op,
        epsilon_pe,
        ec,
        verbose=False,
    ):
        x0 = torch.from_numpy(x0.copy()).float().to(self.device)
        self.x0 = x0
        self.delta_plus = delta_plus
        self.project = search_projection(self.schema, x0.cpu().numpy())
        x = x0.clone()
        x_b, delta_base = self.find_x_boundary(x)
        if x_b is None:
            self.feasible = False
            return None

        x_b = x_b.detach().clone()
        delta_base = delta_base.detach().clone()
        delta = delta_base + delta_plus

        X_feas = self.feasible_set(
            x_b,
            radius=perturb_radius,
            num_samples=num_samples,
            random_state=self.random_state,
        ).float()

        y_feas = self.make_prediction(X_feas).to(self.device)
        if (y_feas == self.y_target).all() or (y_feas != self.y_target).all():
            # The loss contrasts both classes around x_b; one is missing.
            self.feasible = False
            return None

        X_feas_pos = X_feas[y_feas == self.y_target].reshape(
            [sum(y_feas == self.y_target), -1]
        )
        X_feas_neg = X_feas[y_feas == (1 - self.y_target)].reshape(
            [sum(y_feas == (1 - self.y_target)), -1]
        )

        loss_fn = RBRLoss(
            X_feas,
            X_feas_pos,
            X_feas_neg,
            epsilon_op,
            epsilon_pe,
            sigma,
            device=self.device,
        )

        _, x = self.optimize(x_b, delta, loss_fn, verbose=verbose)

        self.feasible = self.make_prediction(x) == self.y_target
        return x.cpu().detach().numpy().squeeze()
