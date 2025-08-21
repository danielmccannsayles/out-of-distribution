# Taken from: https://pastebin.com/iEgQPB8r (cleaned up + video code removed)

# NOTE: if we start getting div by 0 errors, increase nudge_magnitude, or handle this
import math

import torch
import torchvision.models as models
from PIL import Image
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from torchvision import transforms
from torchvision.models import ResNet18_Weights

torch.set_grad_enabled(False)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


# Helper - load, preprocess image
def image_path_to_resnet18_input(image_path):
    # Presumably this is original pre-processing
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    # Convert the image to RGB if it has an alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    input_tensor = preprocess(image)

    # Add a batch dimension (PyTorch expects a batch)
    input_tensor = input_tensor.unsqueeze(0)  # type: ignore

    return input_tensor


# Cosine similarity (in degrees) between two tensors :p
def cossim_deg(tensor1, tensor2):
    # Flatten the tensors
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()

    # Assert that the tensors have the same size
    assert flat_tensor1.shape == flat_tensor2.shape, "Tensors must have the same size"

    # Calculate cosine similarity
    similarity = torch_cosine_similarity(
        flat_tensor1.unsqueeze(0), flat_tensor2.unsqueeze(0), dim=-1
    )

    coss = similarity.item()
    print(f"intermediate: {coss}")

    # Clamp to valid domain for acos to handle floating-point precision errors
    coss = max(-1.0, min(1.0, coss))

    return math.degrees(math.acos(coss))


# Add some noise to a tensor, using random guassian
# Originally was 1e-5 - that was too little of nudging.
def nudge(tensor, nudge_magnitude=1e-4, use_random_gaussian=True):
    original_magnitude = torch.norm(tensor)

    # rand guass else ones
    if use_random_gaussian:
        nudge_tensor = torch.randn_like(tensor)
    else:
        nudge_tensor = torch.ones_like(tensor)

    scaled_nudge_tensor = nudge_tensor * (original_magnitude * nudge_magnitude)
    modified_tensor = tensor + scaled_nudge_tensor
    new_magnitude = torch.norm(modified_tensor)

    scaling_factor = original_magnitude / new_magnitude
    final_tensor = modified_tensor * scaling_factor

    return final_tensor


# Get model & define conv_outputs
resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


def get_conv_outputs(input_tensor):
    # List to store the outputs of all convolutional layers
    conv_outputs = []

    # Hook function to store the output of a convolutional layer
    def hook(module, _, output):
        if isinstance(module, torch.nn.Conv2d):
            conv_outputs.append(output)

    # List to store handles for the hooks
    handles = []

    # Register the hook to all modules within the model and store handles
    for module in resnet18.modules():
        handle = module.register_forward_hook(hook)
        handles.append(handle)

    # Running the forward pass
    with torch.no_grad():
        resnet18(input_tensor)

    # Unregistering the hooks
    for handle in handles:
        handle.remove()

    return conv_outputs


# Presumably this can be anything that maps flat tensors to flat tensors of the same length.
# TODO: try on different levels, e.g. not 3 from end
def black_box(x):
    y = get_conv_outputs(x)[-6]
    return y


# Main func. Nudge tensor, check cossim, run through blackbox, check again
def get_distortion_ratio(point, black_box):
    nudged = nudge(point)
    control = cossim_deg(point, nudged)
    mapped_point = black_box(point)
    mapped_nudged = black_box(nudged)
    test = cossim_deg(mapped_point, mapped_nudged)

    return test / control


paths = [
    "./imgs/train/fish.JPEG",
    "./imgs/train/shark.JPEG",
    "./imgs/validate/snake.JPEG",
    "./imgs/random/me.jpeg",
]

for i in range(len(paths)):
    x = image_path_to_resnet18_input(paths[i])
    r = get_distortion_ratio(x, black_box)
    print(r)
    print("\n\n")
