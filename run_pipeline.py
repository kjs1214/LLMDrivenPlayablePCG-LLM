import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

model = PeftModel.from_pretrained(base_model, "./lora_summary", adapter_name="model1_summary")
model.load_adapter("./lora_json", adapter_name="model2_json")

print("\n" + "="*60)
print("대화를 입력하세요. (종료: 'q')")
print("="*60)

while True:
    player_input = input("\n[플레이어 입력]: ")
    if player_input.lower() in ['q', 'quit', 'exit']:
        break
    if not player_input.strip():
        continue

    # ---------------------------------------------------------
    # [STEP 1] Model 1 
    # ---------------------------------------------------------
    model.set_adapter("model1_summary") 
    
    prompt1 = (
        "<|im_start|>user\n다음 대화를 맵 파라미터로 요약해.\n"
        "[NPC]: 다음 구역으로 이동하기 전에, 앞에 어떤 장소가 펼쳐질 것 같은지 말해줘\n"
        f"[Player]: {player_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs1 = tokenizer(prompt1, return_tensors="pt").to("cuda")
    print("\nModel 1: 플레이어 의도 분석 중...")
    
    with torch.no_grad():
        out1 = model.generate(**inputs1, max_new_tokens=100, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
    
    summary_text = tokenizer.decode(out1[0], skip_special_tokens=False).split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    summary_match = re.search(r'\{.*?\}', summary_text)
    if not summary_match:
        print(f"Error: {summary_text}")
        continue
        
    summary_json_str = summary_match.group(0)
    print(f"[추출된 요약 태그]: {summary_json_str}")

    # ---------------------------------------------------------
    # [STEP 2] Model 2 
    # ---------------------------------------------------------
    model.set_adapter("model2_json")
    
    prompt2 = (
        "<|im_start|>user\n다음 요약을 PCG JSON으로 변환해.\n"
        f"{summary_json_str}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs2 = tokenizer(prompt2, return_tensors="pt").to("cuda")
    print("Model 2: 온톨로지 기반 PCG 언리얼 JSON 생성 중...")
    
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=600, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        
    final_text = tokenizer.decode(out2[0], skip_special_tokens=False).split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    final_match = re.search(r'\{.*\}', final_text, re.DOTALL)
    
    print(final_text)