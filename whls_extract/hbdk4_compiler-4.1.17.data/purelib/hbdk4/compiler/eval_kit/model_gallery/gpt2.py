import torch
from hbdk4.compiler.eval_kit import trace
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


def retrieve_gpt2(sizes, num_layers, dimension_model, num_heads):
    """
    get gpt2-base model.
    """
    config = GPT2Config(
        n_embd=dimension_model,
        n_layer=num_layers,
        n_head=num_heads,
        activation_function="gelu",
        return_dict=False,
    )
    gpt2 = GPT2LMHeadModel(config)

    ei = torch.randint(0, 100, sizes, dtype=torch.long)
    return (
        trace(gpt2, ei, splat=False),
        ei,
    )
