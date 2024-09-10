import torch
from typing import Callable
import tqdm

def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    for epoch in range(1, num_epochs+1):
        model.train()
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        data_iter = tqdm.tqdm(
            enumerate(trainloader),
            desc=f"Epoch {epoch}",
            total=len(trainloader),
            bar_format="{l_bar}{r_bar}"
        )
        for _, batch in data_iter:
            optimizer.zero_grad()
            data = {key: value.to(device) for key, value in batch.items()}
            next_sent_output, mask_lm_output = model(data["bert_input"], data["segment_label"])

            next_loss = loss_function(next_sent_output, data["is_next"])

            mask_loss = loss_function(mask_lm_output.transpose(1, 2), data["bert_label"])
            loss = next_loss + mask_loss
            loss.backward()
            optimizer.step()

            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()
        print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Acc: {total_correct/total_element:.4f}")