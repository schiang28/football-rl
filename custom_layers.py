import torch
from torch import nn

class GNNCommunicationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_agents):
        super().__init__()
        self.n_agents = n_agents

        self.encoder = nn.Linear(input_dim, output_dim)
        self.msg_processor = nn.Linear(output_dim, output_dim)

        self.integrator = nn.Linear(output_dim * 2, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x shape: [batch, n_agents, input_dim]
        h_local = self.activation(self.encoder(x)) # [batch, n_agents, output_dim]

        # subtract agent's own features from global sum for other agent information
        total_sum = torch.sum(h_local, dim=1, keepdim=True) # [batch, 1, output_dim]
        messages = (total_sum - h_local) / (self.n_agents - 1) # [batch, n_agents, output_dim]

        msg_out = self.activation(self.msg_processor(messages))
        combined = torch.cat([h_local, msg_out], dim=-1)
        
        return self.activation(self.integrator(combined))