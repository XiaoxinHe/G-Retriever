from typing import Any
from torch_geometric.data import Batch
import torch
from torch_geometric.utils import mask_to_index, index_to_mask


def batch_subgraph(edge_index,
                   node_ids,
                   num_nodes,
                   num_hops=3,
                   fans_out=(50, 50, 50)
                   ):

    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    row, col = edge_index
    inc_num = 0
    batch_id = 0

    for node_idx in node_ids:
        subsets = [node_idx.flatten()]
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)

        for _ in range(num_hops):
            node_mask.fill_(False)

            node_mask[subsets[-1]] = True
            edge_mask = torch.index_select(node_mask, 0, row)

            neighbors = col[edge_mask]
            if len(neighbors) > fans_out[_]:
                perm = torch.randperm(len(neighbors))[:fans_out[_]]
                neighbors = neighbors[perm]

            subsets.append(neighbors)

        subset, ind = torch.unique(torch.cat(subsets), return_inverse=True)

        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # Relabel Node
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=edge_index.device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx[edge_index_sub]

        # Batching Graph
        edge_index_sub += inc_num

        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num + ind[0].item())
        batch_list.extend([batch_id for _ in range(len(subset))])

        inc_num += len(subset)
        batch_id += 1

    subset, mapping, batch = torch.cat(subset_list), torch.as_tensor(mapping_list), torch.as_tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch


class TAGCollator(object):
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, original_batch):
        mybatch = {}
        for k in original_batch[0].keys():
            mybatch[k] = [d[k] for d in original_batch]

        subset, edge_index_sub, mapping, batch = batch_subgraph(
            edge_index=self.graph.edge_index,
            node_ids=torch.tensor(mybatch['id']),
            num_nodes=self.graph.num_nodes
        )

        mybatch['x'] = self.graph.x[subset]
        mybatch['y'] = self.graph.y[subset]
        mybatch['edge_index'] = edge_index_sub
        mybatch['mapping'] = mapping
        mybatch['batch'] = batch

        return mybatch


class GQACollator(object):
    def __init__(self, graph):
        pass

    def __call__(self, original_batch):
        mybatch = {}
        for k in original_batch[0].keys():
            mybatch[k] = [d[k] for d in original_batch]

        if 'graph' in mybatch:
            mybatch['graph'] = Batch.from_data_list(mybatch['graph'])

        return mybatch


class DefaultCollator(object):
    def __init__(self, graph):
        pass

    def __call__(self, original_batch):
        mybatch = {}
        for k in original_batch[0].keys():
            mybatch[k] = [d[k] for d in original_batch]
        return mybatch


collate_funcs = {
    'graph_llm': GQACollator,
    'gqa_llm': GQACollator,
    'inference_llm':DefaultCollator,
}
