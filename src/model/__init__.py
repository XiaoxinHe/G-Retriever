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
    '7b': 'meta-llama/llama2-7b-hf',
    '13b': 'meta-llama/llama2-13b-hf',
    '7b_chat': 'meta-llama/llama2-7b-chat-hf',
    '13b_chat': 'meta-llama/llama-13b-chat-hf',
}
