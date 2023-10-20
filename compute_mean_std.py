from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def compute_dataset_mean_and_std(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean = 0.
    std = 0.
    num_samples = 0
    for data in loader:
        inputs, _ = data
        batch_samples = inputs.size(0)
        inputs = inputs.view(batch_samples, inputs.size(1), -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return mean, std


mean, std = compute_dataset_mean_and_std(ImageFolder("data", transforms.ToTensor()))
print(mean, std)
