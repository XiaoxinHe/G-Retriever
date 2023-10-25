import math
import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.gnn import load_gnn_model


MAX_NEW_TOKENS = 20
ignore_index = -100


class PromptTuningLLM(torch.nn.Module):

    def __init__(
        self,
        graph,
        instruction,
        args,
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.instruction = instruction
        self.prompt_type = args.llm_prompt_type

        num_virtual_tokens = args.llm_num_virtual_tokens
        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "device_map": "auto",
            "revision": "main",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        # freeze all parameters except prompt embeddings
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        print('Finish loading LLAMA!')

        # graph prompt tuning
        if self.prompt_type == 'graph':
            self.graph_encoder = load_gnn_model[args.gnn_model_name](
                in_channels=graph.x.shape[-1],
                out_channels=4096,
                hidden_channels=args.gnn_hidden_dim,
                num_layers=args.gnn_num_layers,
                dropout=args.gnn_dropout,
            ).to(self.model.device)
            self.graph = graph.to(self.model.device)

        # prompt tuning
        elif self.prompt_type == 'text':
            init_token_ids = self.tokenizer(self.instruction).input_ids
            num_text_tokens = len(init_token_ids)
            if num_text_tokens < num_virtual_tokens:
                num_reps = math.ceil(num_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:num_virtual_tokens]
            word_embeddings = self.model.get_input_embeddings().weight
            self.prompt = torch.nn.Parameter(
                word_embeddings[torch.LongTensor(init_token_ids)].detach().clone().to(torch.float32)).to(
                self.model.device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, ids):
        n_embeds, _ = self.graph_encoder(self.graph.x, self.graph.edge_index)
        inputs_embeds = n_embeds[ids]
        return inputs_embeds

    def forward(self, samples):

        instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)

        # encode special tokens
        pad_embeds = self.model.model.embed_tokens(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        bos_embeds = self.model.model.embed_tokens(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        # encode desc
        model_inputs = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        if self.prompt_type == 'graph':
            # encode graphs
            prompt_embeds = self.encode_graphs(samples['id']).unsqueeze(1)
        elif self.prompt_type == 'text':
            prompt_embeds = self.prompt.repeat(batch_size, 1, 1)

        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.eos_token_id]
            input_ids = model_inputs["input_ids"][i][:self.max_txt_len] + instruction+label_input_ids
            inputs_embeds = self.model.model.embed_tokens(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, prompt_embeds[i], inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [ignore_index] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [ignore_index] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)

        # encode special tokens
        pad_embeds = self.model.model.embed_tokens(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
        bos_embeds = self.model.model.embed_tokens(torch.tensor(self.tokenizer.bos_token_id)).unsqueeze(0)

        # encode desc
        model_inputs = self.tokenizer(samples["desc"], add_special_tokens=False)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        if self.prompt_type == 'graph':
            # encode graphs
            prompt_embeds = self.encode_graphs(samples['id']).unsqueeze(1)
        elif self.prompt_type == 'text':
            prompt_embeds = self.prompt.repeat(batch_size, 1, 1)

        for i in range(batch_size):
            # Add bos & eos token
            input_ids = model_inputs["input_ids"][i][:self.max_txt_len] + instruction
            inputs_embeds = self.model.model.embed_tokens(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, prompt_embeds[i], inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=MAX_NEW_TOKENS,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'], 'pred': pred, 'label': samples['label']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
