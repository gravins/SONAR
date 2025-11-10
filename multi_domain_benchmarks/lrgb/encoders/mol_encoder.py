import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(torch.nn.Module):
    r"""The atom encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = AtomEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """
    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x, pestat):
        encoded_features = 0
        for i in range(x.shape[1]):
            encoded_features += self.atom_embedding_list[i](x[:, i])

        x = encoded_features
        return x


class BondEncoder(torch.nn.Module):
    r"""The bond encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = BondEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """
    def __init__(self, emb_dim: int):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            edge_attr = edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        edge_attr = bond_embedding
        return edge_attr



