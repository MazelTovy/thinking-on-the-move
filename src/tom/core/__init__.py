"""
Core ToM pipeline:
- data_prep: SFT data preparation (cluster pools → personas → train/val JSONL)
- train_lora: LoRA fine-tuning of Llama-3.1-8B / Qwen2.5-7B
- infer_vllm: batch inference via vLLM with parallel candidate-set construction
- prepare_synthetic_unseen: cold-start hold-out experiment (exp03 in poster)
"""
