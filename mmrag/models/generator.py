import torch
from torch import nn
from typing import Optional, List, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

class Generator:
    """
    Text generator using LLaMA/Mistral with LoRA adapter support.
    Can condition on fused embeddings.
    """
    def __init__(
        self,
        model_name: str = 'meta-llama/Llama-2-7b-hf',
        adapter_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        fused_emb: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **gen_kwargs
    ) -> str:
        """
        Generate text conditioned on prompt and (optionally) fused embedding.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if fused_emb is not None:
            # Optionally prepend fused embedding as a special token (simple approach)
            # For more advanced fusion, modify model architecture
            # Here, we just concatenate to input embeddings
            input_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
            fused_emb = fused_emb.unsqueeze(1) if fused_emb.dim() == 2 else fused_emb
            input_embeds = torch.cat([fused_emb, input_embeds], dim=1)
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **gen_kwargs
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **gen_kwargs
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 