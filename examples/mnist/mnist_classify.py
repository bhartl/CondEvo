import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 64
epochs = 5
lr = 0.01

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=batch_size, shuffle=False
)

# Model definition
class SimpleMNISTClassifier(nn.Module):
    def __init__(self, n1=128, n2=5):
        super().__init__()
        self.fc1 = nn.Linear(28*28, n1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n1, n2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(n2, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def serlialize_model(model):
    """ a single tensor """
    return torch.cat([param.view(-1) for param in model.parameters()])


def deserialize_model(model, params):
    """ a single tensor """
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(params[offset:offset + numel].view(param.size()))
        offset += numel
    assert offset == params.numel(), "Mismatch in parameter count during deserialization."


def train_backprop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMNISTClassifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs} completed.')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f'Test accuracy: {100. * correct / total:.2f}%')


def test_serialize_deserialize():
    model = SimpleMNISTClassifier()
    params = serlialize_model(model)
    new_model = SimpleMNISTClassifier()
    deserialize_model(new_model, params)

    # Check if the parameters match
    for param1, param2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(param1.data, param2.data), "Parameters do not match after serialization/deserialization."


# def hades(generations=20, popsize=512, autoscaling=True, sample_uniform=True,
#           is_genetic=False, diffuser="DDIM", tensorboard=False, sharpen_sampling=25,
#           reload=False,
#           ):
#     # define the fitness function
#     targets = [[0.1, 4.0, -3.0],
#                [-2., 0.5, -0.25],
#                [1.0, -1., 1.4],
#                ]
#
#     # targets = [[0.1, 4.0, ],
#     #            [-2., 0.5, ],
#     #            [1.0, -1., ],
#     #            ]
#
#     # targets = [
#     #     [10.332, 20.044, 10.399, ],
#     #     [-10.418, 10.795, -20.232, ],
#     #     [0.897, -10.847, -20.126,],
#     # ]
#     targets = torch.tensor(targets)
#
#     # define the neural network
#     num_params = len(targets[0])
#     mlp = MLP(num_params=num_params, num_hidden=32, num_layers=3, activation='SiLU',
#               batch_norm=True, )
#
#     # from condevo.nn.self_attention import SelfAttentionMLP
#     # mlp = SelfAttentionMLP(num_params=num_params, num_hidden=32, num_layers=6, activation='ReLU',
#     #                         batch_norm=True, dropout=0.1, num_heads=4, num_conditions=0)
#
#     # mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
#     if diffuser == "DDIM":
#         # define the diffusion model
#         diffuser = DDIM(nn=mlp,
#                         num_steps=100,
#                         noise_level=0.5,
#                         autoscaling=autoscaling,
#                         sample_uniform=sample_uniform,
#                         alpha_schedule="linear",
#                         matthew_factor=np.sqrt(0.5),
#                         diff_range=10.0,
#                         # predict_eps_t=True,
#                         log_dir="data/logs/hades" * tensorboard,
#                         normalize_steps=True,
#                         predict_eps_t=False,
#                         )
#     else:
#         # define the fect flow
#         diffuser = RectFlow(nn=mlp,
#                             num_steps=100,
#                             noise_level=0.5,
#                             autoscaling=autoscaling,
#                             sample_uniform=sample_uniform,
#                             matthew_factor=1,#np.sqrt(0.5),
#                             diff_range=10.0,
#                             log_dir="data/logs/hades_RF" * tensorboard,
#                             )
#
#     # define the evolutionary strategy
#     solver = HADES(num_params=num_params,
#                    model=diffuser,
#                    popsize=popsize,
#                    sigma_init=3.0,
#                    is_genetic_algorithm=is_genetic,
#                    selection_pressure=10,
#                    adaptive_selection_pressure=False,            # chose seletion_pressure such that `elite_ratio` individuals have cumulative probability
#                    ###########################################
#                    ## free sampling
#                    # elite_ratio=0.,                 # 0.1 if not is_genetic else 0.4,
#                    # mutation_rate=0.,               # 0.05
#                    # unbiased_mutation_ratio=0.,     # 0.1
#                    # readaptation=False,
#                    ###########################################
#                    ###########################################
#                    ## protect locals
#                    elite_ratio=0.2 if not is_genetic else 0.4,
#                    mutation_rate=0.05,
#                    unbiased_mutation_ratio=0.1,
#                    readaptation=True,
#                    ###########################################
#                    random_mutation_ratio=0.125,
#                    crossover_ratio=0.0,
#                    forget_best=True,
#                    diff_lr=0.003,
#                    diff_optim="Adam",
#                    diff_max_epoch=100,
#                    diff_batch_size=256,
#                    diff_weight_decay=1e-6,
#                    buffer_size=0,  # don't restrict buffer size
#                    diff_continuous_training=reload,
#                    model_path="data/models/hades.pt" * reload,
#                    training_interval=10,
#                    )
#
#     # evolutionary loop
#     x, f = [], []
#     for g in range(generations):
#         x_g = solver.ask()          # sample new parameters
#         f_g = foo(x_g, targets)     # evaluate fitness
#         print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
#         solver.tell(f_g)            # tell the solver the fitness of the parameters
#         x.append(x_g)
#         f.append(f_g)
#
#         if g > sharpen_sampling:
#             solver.model.matthew_factor = 1.0  # first go for diversity, then sharpen sampling after `N` generations
#
#     if num_params == 3:
#         # plotting results
#         plot_3d(x, f, targets=targets)
#
#     else:
#         # plotting results
#         plot_2d(x, f, targets=targets)


def train_hades():
    from condevo.es import HADES
    from condevo.diffusion import DDIM
    from condevo.nn import MLP
    import numpy as np

    classifier = SimpleMNISTClassifier()
    num_params = len(serlialize_model(classifier))

    diffusion_mlp = MLP(num_params=num_params, num_hidden=128, num_layers=6, activation='SiLU', batch_norm=True)
    diffuser = DDIM(nn=diffusion_mlp,
                    num_steps=100,
                    noise_level=0.5,
                    autoscaling=False,
                    sample_uniform=False,
                    alpha_schedule="linear",
                    matthew_factor=np.sqrt(0.5),
                    diff_range=10.0,
                    log_dir="data/logs/hades",
                    normalize_steps=True,
                    predict_eps_t=False)

    # maybe use decoder ? inverse Lindenstrauß | can we make use of network inversion?
    # latent reservoir diffusion evolutionary strategy


if __name__ == "__main__":
    test_serialize_deserialize()