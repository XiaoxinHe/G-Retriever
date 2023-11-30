from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.graph_llm import GraphLLM


load_model = {
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'graph_llm': GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    '7b': 'llama2/llama2_7b_hf',
    '13b': 'llama2/llama2_13b_hf',
    '7b_chat': 'llama2/llama2_7b_chat_hf',
    '13b_chat': 'llama2/llama2_13b_chat_hf',
}
