import torch


def reform_input(inputs, attention_mask = None, ending_token = 2):
    # print("inputs: ", inputs)
    # print("attention_mask: ", attention_mask)
    ## input a tensor of size BSZ x Length
    # print(torch.where(inputs==ending_token))
    max_idx = torch.max(torch.where(inputs==ending_token)[1])
    # print("max_idx: ", max_idx)
    inputs = inputs[:, :max_idx+1]
    # print("inputs_final: ", inputs)
    # print("***********************")
    if attention_mask is not None:
        attention_mask = attention_mask[:, :max_idx+1]

    return inputs, attention_mask