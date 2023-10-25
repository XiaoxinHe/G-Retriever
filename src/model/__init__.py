from src.model.graph_llm import GraphLLM
from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.gqa_llm import GraphQALLM
from src.model.gnn import GCN
from src.model.gnn import GAT


load_model = {
    'graph_llm': GraphLLM,
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'gqa_llm': GraphQALLM,
    'gcn': GCN,
    'gat': GAT,
}


llama_model_path = {
    '7b': '/home/xiaoxin/llama2/llama2_7b_hf',
    '13b': '/home/xiaoxin/llama2/llama2_13b_hf',
    '7b_chat': '/home/xiaoxin/llama2/llama2_7b_chat_hf',
    '13b_chat': '/home/xiaoxin/llama2/llama2_13b_chat_hf',
}
