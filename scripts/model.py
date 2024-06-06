import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, init_randomly : bool = False):
        super().__init__()

        # Create and store the actual model.
        self.embedded_model = Model_(init_randomly)
                

    def forward(self, batch):
        if isinstance(batch, dict) and 'data' in batch:
            logits = self.embedded_model(batch['data'])
            out = {'logits' : logits}
            return out
        else:
            return self.embedded_model(batch)


class Model_(nn.Module):
    """
    A PyTorch model for performing pre-defined convolutional operations on input data.

    Args:
        init_randomly (bool): If True, initialize the weights randomly. If False, initialize the weights with specific values.
    """

    def __init__(self, init_randomly):
        super().__init__()
        
        H = torch.FloatTensor([
            [-1, -1, -1],
            [2, 2, 2],
            [-1, -1, -1],
        ])
        V = torch.FloatTensor([
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1],
        ])
        w = torch.stack([H, V]).unsqueeze(1)
        self.w = nn.Parameter(w, False)

        self.relu = nn.ReLU()

        self.conv_1x1 = nn.Conv2d(2, 1, 1, bias=False)
        if not init_randomly:
            with torch.no_grad():
                self.conv_1x1.weight[0, 0, 0, 0] = 1.0
                self.conv_1x1.weight[0, 1, 0, 0] = -1.0


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = F.conv2d(x, self.w)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = out.sum((1,2,3))
        return out

