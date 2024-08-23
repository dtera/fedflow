from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# model_name_or_path = "facebook/opt-350m"
model_name_or_path = "../data/Llama-2-7b-chat-hf"
output_dir = "../data/outputs/results_modified"
model_dir = "../data/outputs/model"
final_dir = "../data/outputs/final"
train_data_path = '../data/train_data.jsonl'


def peft_train():
    torch.nn.MultiheadAttention(1024, 8, batch_first=True)
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # device_map="auto"
    )
    base_model.config.use_cache = False

    # Training Params
    train_params = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=4e-5,
        weight_decay=0.001,
        # fp16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        max_seq_length=200,
        dataset_text_field="text"
    )

    # LoRA Config
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset('json', data_files={'train': train_data_path}, split='train')
    # Trainer with LoRA configuration
    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=llama_tokenizer,
        args=train_params
    )

    # Training
    fine_tuning.train()
    # Save Model
    fine_tuning.model.save_pretrained(model_dir)
    fine_tuning.save_model(final_dir)


def peft_inference():
    model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    # model = model.to("cuda")
    model.eval()
    inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])


def opt_1_3b():
    generator = pipeline('text-generation', model="facebook/opt-1.3b", do_sample=True)
    generator("What are we having for dinner?")


if __name__ == "__main__":
    peft_train()
    # peft_inference()
    # opt_1_3b()
