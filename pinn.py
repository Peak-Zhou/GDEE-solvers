import numpy as np
import torch

# ------- Selecting：MLP or KAN ----------
USE_KAN = False

if USE_KAN:
    from kan import KAN
    class KANWrapper(torch.nn.Module):
        def __init__(self, input_dim=2, output_dim=1, width=16, grid=21):
            super().__init__()
            width_list = [input_dim, width, output_dim]
            self.kan = KAN(
                width=width_list,
                grid=grid
            )
        def forward(self, x):
            return self.kan(x)
else:
    class MLP(torch.nn.Module):
        def __init__(self, layers=[2, 64, 64, 64, 64, 1]):
            super().__init__()
            modules = []
            for i in range(len(layers)-2):
                modules.append(torch.nn.Linear(layers[i], layers[i+1]))
                modules.append(torch.nn.Tanh())
            modules.append(torch.nn.Linear(layers[-2], layers[-1]))
            self.net = torch.nn.Sequential(*modules)
        def forward(self, x):
            return self.net(x)

def pinn_func(x0, h, xmin, xmax, tmin, tmax, ut0, ut0x, xpnt, xdot, idx=0,
              epochs=100000, N=10000, N1=1000, lr=0.001, device=None, use_kan=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"【PINN】Case {idx}: device：{device}, network：{'KAN' if use_kan else 'MLP'}")

    def setup_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    setup_seed(888888)

    def interior(n=N):
        t = tmin + (tmax-tmin)*torch.rand(n, 1, device=device)
        x = xmin + (xmax-xmin)*torch.rand(n, 1, device=device)
        cond = torch.zeros_like(x)
        return t.requires_grad_(True), x.requires_grad_(True), cond

    def left(n=N1):
        x = xmin + (xmax-xmin)*torch.rand(n, 1, device=device)
        t = tmin * torch.ones_like(x, device=device)
        cond = ut0 * ut0x(x)
        cond = cond.to(torch.float)
        return t.requires_grad_(True), x.requires_grad_(True), cond

    if use_kan:
        u = KANWrapper().to(device)
    else:
        u = MLP().to(device)

    loss = torch.nn.MSELoss()

    def gradients(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                       create_graph=True,
                                       only_inputs=True, )[0]
        else:
            return gradients(gradients(u, x), x, order=order-1)

    def l_interior(u):
        t, x, cond = interior()
        uxt = u(torch.cat([t, x], dim=1))
        a = xdot(t)
        return loss(gradients(uxt, t, 1) + a * gradients(uxt, x, 1), cond)

    def l_left(u):
        t, x, cond = left()
        uxt = u(torch.cat([t, x], dim=1))
        return loss(uxt, cond)

    opt = torch.optim.Adam(params=u.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50000, gamma=0.5)
    loss_history = []

    for epoch in range(epochs):
        opt.zero_grad()
        l = 0.01*l_interior(u) + l_left(u)
        l.backward()
        opt.step()
        scheduler.step()
        loss_history.append(l.item())
        if epoch % 2000 == 0 or epoch == epochs - 1:
            print(f"[Case {idx}] Epoch {epoch}: Loss={l.item():.6e}")

    loss_array = np.column_stack((np.arange(epochs), np.array(loss_history)))
    np.savetxt(f'pinn_loss_history_{idx}.txt', loss_array, fmt=['%d', '%.8e'], header='epoch\tloss')
    # torch.save(u.state_dict(), f"pinn_model_{idx}.pt")
    return u
