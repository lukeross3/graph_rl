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
        n_nodes: int,
        d_model: int = 32,
        d_emb: int = 32,
        n_heads: int = 4,
        dim_feedforward: int = 2048,
        n_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize the Model

        Args:
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

        # Store dimensionalities
        self.d_emb = d_emb

        # Initialize embedding layer
        self.embeddings = nn.EmbeddingBag(n_nodes, d_emb, mode="sum")

        # Initialize transformer model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, graph: Graph) -> torch.Tensor:
        """Generate probability distribution over processor assignments from graph

        Args:
            graph (Graph): Graph for which to generate processor assignment

        Returns:
            torch.Tensor: (n x p)-dimensional tensor where n is the number of nodes and p is the
                number of processors. Produces a probability distribution over processors for each
                node.
        """

        # Get embeddings from EmbeddingBag
        inputs = []
        offsets = []
        for node in graph.nodes:
            offsets.append(len(inputs))
            inputs.extend(node.input_dependencies)
        inputs = torch.LongTensor(inputs)
        offsets = torch.LongTensor(offsets)
        embs = self.embeddings(inputs, offsets)
        embs = torch.unsqueeze(embs, 1)

        print("embs: ", embs)

        # Run through transformer
        out = self.transformer_encoder(embs)

        print("out: ", out)  # Why is this nan?

        # Compute softmax
        probs = F.softmax(out, dim=1)

        print("probs: ", probs)

        return probs


class TFDecoderWithEmbs(nn.Module):
    pass


import torch

model = TFEncoderWithEmbs(6)
print(count_parameters(model))

# Import MPI
from mpi4py import MPI
from graph_rl import AddN

# Init the graph
nodes = [AddN(1) for _ in range(4)]
connections = [
    (-1, 0),
    (0, 1),
    (1, 2),
    (2, 3),
]
comm = MPI.COMM_WORLD
g = Graph(nodes, connections, comm)
y = model(g)
