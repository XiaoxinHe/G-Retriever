from src.dataset.expla_graphs import ExplaGraphsDataset
from src.dataset.scene_graphs import SceneGraphsDataset
from src.dataset.scene_graphs_baseline import SceneGraphsBaselineDataset
from src.dataset.webqsp import WebQSPDataset
from src.dataset.webqsp_baseline import WebQSPBaselineDataset


load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'scene_graphs': SceneGraphsDataset,
    'scene_graphs_baseline': SceneGraphsBaselineDataset,
    'webqsp': WebQSPDataset,
    'webqsp_baseline': WebQSPBaselineDataset,
}
