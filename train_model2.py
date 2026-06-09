import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Model loading..")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
base_model = prepare_model_for_kbit_training(base_model)

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(base_model, peft_config)

dataset = load_dataset("json", data_files="data2.json", split="train")

training_args = TrainingArguments(
    output_dir="./lora_json",  
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    max_steps=150,                     
    logging_steps=10,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model, train_dataset=dataset, peft_config=peft_config,
    max_seq_length=1024, tokenizer=tokenizer, args=training_args, dataset_text_field="text"
)

print("Started Training")
trainer.train()
trainer.model.save_pretrained("./lora_json")
print("Ended Training(./lora_json saved)")