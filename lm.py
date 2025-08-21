# alright let's see if we can recreate the llm version of this, done by vooogel: https://x.com/voooooogel/status/1688730813746290688
import math
import textwrap

import numpy as np
import torch
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# torch.set_grad_enabled(False)
# torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)


# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")


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

    # Clamp to valid domain for acos to handle floating-point precision errors
    coss = max(-1.0, min(1.0, coss))

    return math.degrees(math.acos(coss))


# Add some noise to a tensor, using random guassian
# Originally was 1e-5 - that was too little of nudging.
def nudge(tensor, nudge_magnitude=1e-2, use_random_gaussian=True):
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


# Main func. Nudge tensor, check cossim, run through blackbox, check again
def get_distortion_ratio(point, black_box):
    nudged = nudge(point)
    control = cossim_deg(point, nudged)
    mapped_point = black_box(point)
    mapped_nudged = black_box(nudged)
    test = cossim_deg(mapped_point, mapped_nudged)

    return test / control


def ood(s):
    tokenized = tokenizer(s, return_tensors="pt")

    def f(embeds):
        return gpt2(
            inputs_embeds=embeds,
            attention_mask=tokenized.attention_mask,
            return_dict=True,
        ).logits

    base_embeds = gpt2.transformer.wte(tokenized.input_ids)
    scores = [get_distortion_ratio(base_embeds, f) for _ in range(50)]
    print(
        f"{textwrap.shorten(s, width=30).ljust(30)} :: {np.mean(scores): .5f} ({np.std(scores):.5f})"
    )


if __name__ == "__main__":
    ood("Eggplants are purple.")
    ood("Eggplants are fire-engine red.")
    ood(
        "Alan turing theorized that computers would one day be the most powerful machines on the planet"
    )
    ood("lsadfhl; wiffle jfdalkgy burh lgoi")
    ood(
        "Flowers, also known as blooms and blossoms, are the reproductive structures of flowering plants"
    )
    ood(
        "Flowers, also known as blooms and blossoms, are the reproductive structures of flowering elephants that jump"
    )
