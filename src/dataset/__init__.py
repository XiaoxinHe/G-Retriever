from src.dataset.gqa import GQADataset
from src.dataset.gqa_baseline import GQABaselineDataset
from src.dataset.medqa import MedQADataset
from src.dataset.expla_graphs import ExplaGraphsDataset
from src.dataset.expla_graphs_baseline import ExplaGraphsBaselineDataset
from src.dataset.bioasq import BioASQDataset
from src.dataset.bioasq_baseline import BioASQBaselineDataset


load_dataset = {
    'gqa': GQADataset,
    'gqa_baseline': GQABaselineDataset,
    'medqa': MedQADataset,
    'expla_graphs': ExplaGraphsDataset,
    'expla_graphs_baseline': ExplaGraphsBaselineDataset,
    'bioasq': BioASQDataset,
    'bioasq_baseline': BioASQBaselineDataset
}
