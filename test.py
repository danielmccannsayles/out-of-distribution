import math

import cv2
import torch
import torchvision.models as models
from PIL import Image
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from torchvision import transforms
from torchvision.models import ResNet18_Weights

torch.set_grad_enabled(False)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


# Cosine similarity between two tensors :p
def cossim(tensor1, tensor2):
    # Flatten the tensors
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()

    # Assert that the tensors have the same size
    assert flat_tensor1.shape == flat_tensor2.shape, "Tensors must have the same size"

    # Calculate cosine similarity
    similarity = torch_cosine_similarity(
        flat_tensor1.unsqueeze(0), flat_tensor2.unsqueeze(0), dim=-1
    )

    return similarity.item()


def cossim_in_degrees(a, b):
    return math.degrees(math.acos(cossim(a, b)))


# Add some noise to a tensor, using random guassian
def nudge(tensor, nudge_magnitude=1e-5, use_random_gaussian=True):
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


# Nudge. See diff between nudge & orig
def get_distortion_ratio(point, black_box):
    nudged = nudge(point)
    control = cossim_in_degrees(point, nudged)
    mapped_point = black_box(point)
    mapped_nudged = black_box(nudged)
    test = cossim_in_degrees(mapped_point, mapped_nudged)
    return test / control


def get_conv_outputs(input_tensor):
    # List to store the outputs of all convolutional layers
    conv_outputs = []

    # Hook function to store the output of a convolutional layer
    def hook(module, input, output):
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


def image_path_to_resnet18_input(image_path):
    # Define the preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load the image from the provided path
    image = Image.open(image_path)

    # Convert the image to RGB if it has an alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Preprocess the image
    input_tensor = preprocess(image)

    # Add a batch dimension (since PyTorch expects a batch)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor


def generalized_outer_product(vector_array):
    alphabet = "ijklmnopqrstuvwxyzabcdefgh"
    einsum_delimiter = ", "
    einsum_args = ""
    einsum_result = ""
    for i in range(len(vector_array)):
        einsum_args += alphabet[i]
        if i != len(vector_array) - 1:
            einsum_args += einsum_delimiter
        einsum_result += alphabet[i]
    einsum_equation = einsum_args + " -> " + einsum_result
    return torch.einsum(einsum_equation, vector_array)


# expects an array of matrices of shape (r,*) (an array of vectors of any size is okay too, it implies r=1)
def unfurl_rank_r_tensor(matrix_array):
    def conditional_unsqueeze(suspected_vec):
        if len(suspected_vec.size()) < 2:
            return torch.unsqueeze_copy(suspected_vec, 0)
        return suspected_vec

    if len(matrix_array) == 0:
        return torch.tensor([])
    rank = conditional_unsqueeze(matrix_array[0]).size()[0]
    for matrix in matrix_array:
        assert matrix.size()[0] == rank
    target_shape = [m.size()[1] for m in matrix_array]
    unfurled_tensor = torch.zeros(target_shape)
    for i in range(rank):
        outer_product_vector_array = [m[i] for m in matrix_array]
        unfurled_tensor += generalized_outer_product(outer_product_vector_array)
    return unfurled_tensor


def frame_to_resnet18_input(frame):
    # Convert the frame to a PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor


resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


# Presumably this can be anything that maps flat tensors to flat tensors of the same length.
def black_box(x):
    y = get_conv_outputs(x)[-3]
    return y


paths = ["pfp.png", "pfp-m1.png", "pfp-m2.png", "pfp-m3.png", "lizard.jpeg"]

for i in range(len(paths)):
    x = image_path_to_resnet18_input(paths[i])
    r = get_distortion_ratio(x, black_box)
    print(r)
    print("\n\n")


"""

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    input_tensor = frame_to_resnet18_input(frame)

    # Get the number from the CNN
    number = get_distortion_ratio(input_tensor, black_box)

    # Draw the number on the frame
    cv2.putText(frame, str(number), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()

"""
