import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from torchinfo import summary
from tqdm import tqdm
from collections import Counter


class MLPExpert(nn.Module):
    """
    A simple MLP expert for featurizing MNIST latents.
    """
    def __init__(self, input_dim=784, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class ConvExpert(nn.Module):
    """
    A convolutional expert for featurizing MNIST latents.
    """
    def __init__(self, hidden_channels=32, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(hidden_channels, 1, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
    
    def forward(self, x):  # (B, 784)
        x = x.view(x.size(0), 1, 28, 28)
        x = self.net(x)
        x = x.view(x.size(0), 784)
        return x


class GatingNetwork(nn.Module):
    """
    Gating network that outputs a distribution over experts.
    """
    def __init__(self, input_dim=784, hidden_dim=1024, num_experts=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_experts)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Returns softmax logits for gating, shape = [batch_size, num_experts].
        """
        logits = self.net(x)
        gating_weights = self.softmax(logits)
        return gating_weights


class SimpleSGLR(nn.Module):
    """
    Simplistic version of Self-Gated Latent Recurrence for MNIST
    """
    def __init__(self, input_dim=784, hidden_dim=1024, num_experts=4, num_classes=10):
        super().__init__()
        self.num_experts = num_experts
        self.initial_featurizer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # Split experts between MLPExpert and ConvExpert
        number_of_mlp_experts = num_experts // 2
        number_of_conv_experts = num_experts - number_of_mlp_experts
        self.experts = nn.ModuleList([])
        for _ in range(number_of_mlp_experts):
            self.experts.append(MLPExpert(input_dim, hidden_dim))
        for _ in range(number_of_conv_experts):
            self.experts.append(
                ConvExpert(
                    kernel_size=random.choice([3, 5, 7]),
                    stride=1,
                    dilation=random.choice([1, 2, 3])
                )
            )
        
        # Gating network uses num_experts+1 for "exit"
        self.gating_network = GatingNetwork(hidden_dim=hidden_dim * 2, num_experts=len(self.experts) + 1)
        self.final_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Returns:
          - logits (B, num_classes)
          - expert_usage (vector of length num_experts+1, averaged across batch)
          - usage_order (list of ints for each step in the mixture)
          - final_features (B, 784) - the final representation before classifier
        """
        expert_usage, usage_order = [], []
        x = self.initial_featurizer(x)

        # Repeated gating steps (residual usage)
        for _ in range(self.num_experts * 2):
            gating_weights = self.gating_network(x)
            # Average gating weights over batch (if batch_size=1, same as per-sample)
            avg_suggestion = gating_weights.mean(0)
            expert_usage.append(avg_suggestion)
            
            top_choice = avg_suggestion.argmax().item()
            usage_order.append(top_choice)

            # If top_choice == len(self.experts), that means "exit" early
            if top_choice == len(self.experts):
                break

            # Residual connection
            x = self.experts[top_choice](x) + x

        # Average the usage vectors to a single vector
        expert_usage = torch.stack(expert_usage, dim=0).mean(0)

        logits = self.final_classifier(x)
        return logits, expert_usage, usage_order, x


def load_balancing_loss(gating_weights):
    # gating_weights is shape [num_experts+1]
    num_experts_plus_exit = gating_weights.shape[0]
    uniform_distribution = torch.ones(num_experts_plus_exit, device=gating_weights.device) / num_experts_plus_exit
    lb_loss = F.mse_loss(gating_weights, uniform_distribution)
    return lb_loss


def train_one_epoch(model, train_loader, optimizer, device, lb_coef=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        # Flatten images into vectors of size 784
        images = images.view(images.size(0), -1)
        
        optimizer.zero_grad()

        # Forward pass
        logits, gating_weights, _, _ = model(images)
        
        # Main classification loss
        ce_loss = criterion(logits, labels)

        # Load balancing loss (over the average gating vector)
        lb_loss_val = load_balancing_loss(gating_weights)
        total_loss = ce_loss + lb_coef * lb_loss_val

        total_loss.backward()
        optimizer.step()

        running_loss += ce_loss.item() * images.size(0)

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Update progress bar with losses
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'CE Loss': f'{ce_loss.item():.4f}',
                'LB Loss': f'{lb_loss_val.item():.4f}'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device, eval_steps=1000):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)

            logits, _, _, _ = model(images)
            loss = criterion(logits, labels)

            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx >= eval_steps:
                break

    epoch_loss = test_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def plot_digit_usage_pattern(model, device, max_samples=2000):
    """
    Use batch_size=1 to infer the model's usage order on the test set.
    Then, for each digit, we'll collect the usage orders and plot
    a histogram of the top usage sequences.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()

    # We'll keep track of usage_order for each digit
    digit_to_orders = {d: [] for d in range(10)}
    count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Flatten
            images = images.view(images.size(0), -1)

            # Forward pass with batch_size=1
            logits, _, usage_order, _ = model(images)

            digit_label = labels.item()
            digit_to_orders[digit_label].append(tuple(usage_order))  # store as a tuple for hashing
            count += 1

            if count >= max_samples:
                break
    
    # Now we can look at the most common usage sequences for each digit
    fig, axs = plt.subplots(2, 5, figsize=(18, 9))
    axs = axs.flatten()
    for d in range(10):
        ax = axs[d]
        all_orders = digit_to_orders[d]
        # Count them
        c = Counter(all_orders)
        # We only show the top 10 sequences to keep it readable
        most_common = c.most_common(10)

        # We'll make a bar chart of the frequencies of these top 5
        # x-axis is the usage-order string, y is the count
        seq_strings = [str(seq) for (seq, _) in most_common]
        freq_values = [count_ for (_, count_) in most_common]

        ax.bar(range(len(most_common)), freq_values, color='skyblue')
        ax.set_xticks(range(len(most_common)))
        ax.set_xticklabels(seq_strings, rotation=40, ha='right')
        ax.set_title(f"Digit {d}\nTop 10 usage orders")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig('digit_usage_patterns.png')
    plt.close()


def main():
    # Hyperparameters
    batch_size = 32
    num_experts = 8
    hidden_dim = 512
    epochs = 2
    learning_rate = 1e-4
    eval_steps = 1000
    lb_coef = 0.1  # Weight for load balancing loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MNIST dataset (for training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Model, optimizer
    model = SimpleSGLR(
        input_dim=784,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_classes=10
    ).to(device)

    # Print summary
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=epochs
    )

    # Lists to store metrics for plotting
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    epochs_trained = 0

    # Early stopping setup
    best_test_acc = 0
    patience = 5
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, lb_coef=lb_coef)
        test_loss, test_acc = evaluate(model, test_loader, device, eval_steps=eval_steps)

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        epochs_trained = epoch

        print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  [Test]  Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step()

        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                model.load_state_dict(best_model_state)
                break

    print("Training complete!")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    epochs_range = range(1, epochs_trained + 1)
    ax1.plot(epochs_range, train_losses, label='Train Loss')
    ax1.plot(epochs_range, test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, label='Train Accuracy')
    ax2.plot(epochs_range, test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    plot_digit_usage_pattern(model, device, max_samples=10000)


if __name__ == "__main__":
    main()
