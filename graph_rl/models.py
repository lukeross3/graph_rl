import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_rl.graph import Graph


def count_parameters(model: nn.Module) -> int:
    """Get the number of trainable parameters of a pytorch model

    Args:
        model (nn.Module): Pytorch model

    Returns:
        int: number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TFEncoderWithEmbs(nn.Module):
    def __init__(
        self,
        n_procs: int,
        d_model: int = 32,
        n_heads: int = 4,
        dim_feedforward: int = 2048,
        n_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize the Model

        Args:
            n_procs (int): the number of processors available.
            d_model (int, optional): the number of expected features in the input. Defaults to 32.
            n_heads (int, optional): the number of heads in the multiheadattention models. Default
                to 4.
            dim_feedforward (int, optional): the dimension of the feedforward network model.
                Defaults to 2048.
            n_layers (int, optional): the number of sub-encoder-layers in the encoder. Defaults to
                4.
            dropout (float, optional): the dropout value. Defaults to 0.1.
            activation (str, optional): the activation function of intermediate layer, relu or gelu.
                Defaults to "relu".
        """
        super(TFEncoderWithEmbs, self).__init__()

        # Initialize embedding layer
        self.embeddings = nn.EmbeddingBag(n_procs, d_model, mode="sum")

        # Initialize transformer model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Initialize output linear layer
        self.linear = nn.Linear(in_features=d_model, out_features=n_procs)

    def forward(self, graph: Graph, batch_size: int = 1) -> torch.Tensor:
        """Generate probability distribution over processor assignments from graph

        Args:
            graph (Graph): Graph for which to generate processor assignment distributions
            batch_size (int, optional): Number of different distributions to run on the graph. Note
                that if dropout is not enabled (e.g. model is in eval mode) then all generated
                distributions will be identical. Defaults to 1.

        Returns:
            torch.Tensor: (n_nodes X batch_size X n_procs)-dimensional tensor. Produces a
                probability distribution over processors for each node. For an output tensor T,
                sum(T[i][j][:]) == 1.
        """

        # Get embeddings from EmbeddingBag
        inputs = []
        offsets = []
        for node in graph.nodes:
            offsets.append(len(inputs))

            # Edge case for nodes requiring the first input
            if -1 in node.input_dependencies:
                tmp = copy.copy(node.input_dependencies)  # Make a copy to avoid mutating the node
                tmp.remove(-1)
                inputs.extend(tmp)
            else:
                inputs.extend(node.input_dependencies)

        # Cast to Tensor and get embeddings
        # SHAPE: (n_nodes X batch X d_model)
        inputs = torch.LongTensor(inputs)
        offsets = torch.LongTensor(offsets)
        embs = self.embeddings(inputs, offsets)
        embs = torch.unsqueeze(embs, 1)
        embs = embs.repeat(1, batch_size, 1)

        # Run through transformer
        # SHAPE: (n_nodes X batch X d_model)
        tf_out = self.transformer_encoder(embs)

        # Linear layer to cast output to right number of classes (n_procs)
        # SHAPE: (n_nodes X batch X n_procs)
        out = self.linear(tf_out)

        # Compute softmax
        probs = F.softmax(out, dim=2)

        return probs
