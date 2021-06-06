import torch.nn as nn

class ProposalGenerationHead(nn.Module):

    def __init__(self, d_model_list, kernel_size, dout_p):
        super(ProposalGenerationHead, self).__init__()
        assert kernel_size % 2 == 1, 'It is more convenient to use odd kernel_sizes for padding'
        conv_layers = []
        in_dims = d_model_list[:-1]
        out_dims = d_model_list[1:]
        N_layers = len(d_model_list) - 1

        for n, (in_d, out_d) in enumerate(zip(in_dims, out_dims)):
            if n == 0:
                conv_layers.append(nn.Conv1d(in_d, out_d, kernel_size, padding=kernel_size//2))
            else:
                conv_layers.append(nn.Conv1d(in_d, out_d, kernel_size=1))

            if n < (N_layers - 1):
                if dout_p > 0:
                    conv_layers.append(nn.Dropout(dout_p))
                conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        # (B, D, S) <- (B, S, D)
        x = x.permute(0, 2, 1)
        # (B, d, S) <- (B, D, S)
        x = self.conv_layers(x)
        # (B, S, d) <- (B, d, S)
        x = x.permute(0, 2, 1)
        # x = self.fc_layer(x)
        return x
