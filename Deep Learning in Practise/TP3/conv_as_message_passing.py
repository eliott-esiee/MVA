import torch
import torch_geometric


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    C, H, W = image.shape
    # pixel -> node, (C, H, W) -> (H*W, C)
    x = image.permute(1, 2, 0).reshape(-1, C)

    row, col = [], []
    edge_attr_list = []

    # each pixel (i,j)
    for i in range(H):
      for j in range(W):
          center_idx = i * W + j
          # all neighbors in a 3x3 window
          for di in [-1, 0, 1]:
              for dj in [-1, 0, 1]:
                  ni, nj = i + di, j + dj
                  if 0 <= ni < H and 0 <= nj < W:
                      neighbor_idx = ni * W + nj
                      # the pixel at neighbor_idx sends a message to the pixel at center_idx
                      row.append(neighbor_idx)
                      col.append(center_idx)
                      edge_attr_list.append([di, dj])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

    return torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    # reshape (H*W, C) -> (C, H, W)
    C = data.shape[1]
    image = data.reshape(height, width, C).permute(2, 0, 1).contiguous()
    return image


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # <TO IMPLEMENT>
        # Don't forget to call the parent constructor with the correct aguments
        # super().__init__(<arguments>)
        # </TO IMPLEMENT>
        super().__init__(aggr='add')
        self.weight = conv2d.weight  # (out_channels, in_channels, kH, kW)
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size  # (kH, kW)
        self.padding = conv2d.padding          # (pad_h, pad_w)

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """
        pad_h, pad_w = self.padding
        kH, kW = self.kernel_size

        # compute flattened kernel index for each offset (di, dj)
        kernel_row = edge_attr[:, 0] + pad_h  # (E,)
        kernel_col = edge_attr[:, 1] + pad_w  # (E,)
        indices = kernel_row * kW + kernel_col  # (E,)

        # reshape (out_channels, in_channels, kH, kW) -> (kH*kW, out_channels, in_channels)
        W = self.weight.view(self.out_channels, self.in_channels, -1).permute(2, 0, 1)

        # kernel slice for each edge
        W_edge = W[indices]  # (E, out_channels, in_channels)

        # multiply source node features with kernel slice
        x_j_unsq = x_j.unsqueeze(2)  # (E, in_channels, 1)
        message = torch.bmm(W_edge, x_j_unsq).squeeze(2)  # (E, out_channels)

        return message
