import os
import pandas as pd
import torch
import evaluate
import wandb

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, DatasetDict, Dataset, Audio
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainer
from typing import Dict, Union, Any, Optional, Tuple, List
#from transformers import DataCollatorForSeq2SeqWithPadding
#from transformers import DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from peft import LoraConfig, get_peft_model


# --- Configuration ---
# SET THESE PATHS CORRECTLY
DATA_DIR = "./RADDROMUR_22.09" # Directory containing metadata.tsv and the audio folder
AUDIO_FOLDER = "speech" # The actual subfolder name holding the .flac files
METADATA_FILENAME = "metadata.tsv"
CACHE_DIR = "./cache_raddromur" # Optional cache directory

TARGET_LANGUAGE = "is" # ISO 639-1 code for Icelandic
MODEL_ID = "language-and-voice-lab/whisper-large-icelandic-30k-steps-1000h"
TASK = "transcribe"
TEST_SPLIT_SIZE = 0.1 # Use 10% of data for testing

# --- Load Metadata ---
metadata_path = os.path.join(DATA_DIR, METADATA_FILENAME)
#df = pd.read_csv(metadata_path, sep='\t')
# Ensure we load 'filename' and 'podcast_id' along with 'sentence_norm'
df = pd.read_csv(metadata_path, sep='\t', usecols=['filename', 'podcast_id', 'sentence_norm'])

# --- Function to Construct Correct Audio Path ---
def construct_audio_path(row):
    try:
        podcast_id_parts = row['podcast_id'].split('_')
        # *** This line is crucial ***
        podcast_name_dir = "_".join(podcast_id_parts[:-1]) # e.g., 'i_ljosi_sogunnar'
        if not podcast_name_dir:
             podcast_name_dir = row['podcast_id']

        # Construct the nested path
        full_path = os.path.join(DATA_DIR, AUDIO_FOLDER, podcast_name_dir, row['podcast_id'], row['filename'])
        # *** Add this print statement for debugging ***
        # print(f"Debug Path Construction: Base='{DATA_DIR}', Speech='{SPEECH_FOLDER}', NameDir='{podcast_name_dir}', IdDir='{row['podcast_id']}', File='{row['filename']}' -> Result='{full_path}'")
        return full_path
    except Exception as e:
        print(f"Error constructing path for row: {row}. Error: {e}")
        return None
    
# --- Apply the function to create the 'audio_full_path' column ---
print("Constructing audio file paths...")
df['audio_full_path'] = df.apply(construct_audio_path, axis=1)

# --- Create Full Audio Paths --- OLD METHOD ---
#df['audio_full_path'] = df['filename'].apply(lambda fname: os.path.join(DATA_DIR, AUDIO_FOLDER, fname))

# --- Handle potential errors and Rename Transcription Column ---
df = df.dropna(subset=['audio_full_path']) # Remove rows where path construction failed
df = df.rename(columns={"sentence_norm": "transcription"})

# --- Select Relevant Columns for Dataset ---
# We only need the constructed path and the transcription now
df_final = df[['audio_full_path', 'transcription']]

# --- Select Relevant Columns and Convert to Dataset ---
df_subset = df[['audio_full_path', 'transcription']]

# --- Create a SINGLE Dataset object from Pandas ---
full_dataset = Dataset.from_pandas(df_final)
print(f"Created initial Dataset from Pandas DataFrame with {len(full_dataset)} rows.")

# --- Cast Audio Column & Verify Sampling Rate ---
# Cast the column in the Dataset object
full_dataset = full_dataset.cast_column("audio_full_path", Audio(sampling_rate=16000))
# Rename the audio column for consistency
full_dataset = full_dataset.rename_column("audio_full_path", "audio")
print("Audio column cast and renamed.")

# --- Split the Dataset object into Train/Test ---
print(f"Splitting dataset into train/test ({1-TEST_SPLIT_SIZE:.0%}/{TEST_SPLIT_SIZE:.0%})...")
split_dataset_dict = full_dataset.train_test_split(test_size=TEST_SPLIT_SIZE, shuffle=True, seed=42)

# Assign the resulting DatasetDict to your 'dataset' variable
dataset = split_dataset_dict

print("Dataset structure after splitting:", dataset)
# Check an example path to ensure it looks correct
if len(dataset["train"]) > 0:
    print("Example training audio path:", dataset["train"][0]["audio"]["path"])

# --- Load Whisper Processor ---
print("Loading Processor...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=TARGET_LANGUAGE, task=TASK)
print("Processor Loaded.")

print("Defining preprocessing function...")
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch
print("Preprocessing function defined.")

print("Applying preprocessing...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=64) # Adjust num_proc, testing 64 on dgx
print("Preprocessing applied.")


# --- NEW: Step 4.5 - Filter out long sequences --- after error on 449 tokens, but max 448
MAX_LABEL_LENGTH = 448 # Whisper's max decoder length

def filter_long_labels(batch):
    # Check the length of the 'labels' column (token IDs)
    return len(batch["labels"]) <= MAX_LABEL_LENGTH

print(f"Filtering dataset to keep labels with length <= {MAX_LABEL_LENGTH}...")
original_lengths = {split: len(dataset[split]) for split in dataset.keys()}

dataset = dataset.filter(filter_long_labels, num_proc=8) # Use multiple processes

filtered_lengths = {split: len(dataset[split]) for split in dataset.keys()}

print("Filtering complete.")
for split in original_lengths:
    print(f"  {split}: {original_lengths[split]} -> {filtered_lengths[split]} samples remaining.")

print("Defining Data Collator...")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("Data Collator defined.")

print("Loading WER metric...")
metric = evaluate.load("wer")
print("WER metric loaded.")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

print("Loading base model for LoRA...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, # Use bfloat16 on A100
    low_cpu_mem_usage=True,
    #use_flash_attention_2=True, # Enable Flash Attention 2
    attn_implementation="sdpa", # Use SDPA as I couldn't install Flash Attention 2 (or took too long) 
)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

print("Applying LoRA configuration...")
model = get_peft_model(model, config)
model.print_trainable_parameters()
print("Model ready for LoRA fine-tuning with sdpa attention.")
#model.gradient_checkpointing_enable()
#print("Gradient checkpointing enabled.")


# --- Calculate Training Steps (for single GPU) ---
num_train_samples = len(dataset["train"])
#NUM_GPUS = 1 # Explicitly set to 1 as we target a single GPU
NUM_GPUS = torch.cuda.device_count() # Get number of GPUs visible via CUDA_VISIBLE_DEVICES
print(f"--- Detected {NUM_GPUS} GPUs by torch.cuda.device_count() ---")
if NUM_GPUS != 3:
    print(f"*** WARNING: Expected 3 GPUs based on CUDA_VISIBLE_DEVICES, but detected {NUM_GPUS}. Check environment setup. ***")
    # If you see this warning, ensure your environment is set up correctly with 5 GPUs.
PER_DEVICE_BATCH_SIZE = 4 # Adjust based on GPU 3's VRAM usage (A100 should handle this fine)
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 3

effective_batch_size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS
steps_per_epoch = num_train_samples // effective_batch_size
total_steps = steps_per_epoch * NUM_EPOCHS
eval_save_steps = steps_per_epoch // 2 # Example: Evaluate/save twice per epoch

print(f"Num GPUs: {NUM_GPUS}, Per-Device BS: {PER_DEVICE_BATCH_SIZE}, Accum Steps: {GRAD_ACCUM_STEPS}")
print(f"Effective Batch Size: {effective_batch_size}")
print(f"Num Train Samples: {num_train_samples}, Epochs: {NUM_EPOCHS}")
print(f"Calculated Max Steps: {total_steps}")
print(f"Calculated Eval/Save Steps: {eval_save_steps}")

# --- Training Arguments ---
WANDB_PROJECT_NAME = "whisper-lvl-base-raddromur-lora"

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-lvl-base-raddromur-lora",
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    optim="adamw_8bit",
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=int(total_steps * 0.1),
    max_steps=total_steps,

    # --- Precision ---
    bf16=True, # Use BFloat16 on A100
    bf16_full_eval=True,

    # --- Evaluation and Saving ---
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE * 2,
    predict_with_generate=True,
    generation_max_length=225,
    eval_strategy="steps",
    eval_steps=eval_save_steps,
    save_strategy="steps",
    save_steps=eval_save_steps,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=3,

    # --- Logging ---
    logging_steps=max(1, eval_save_steps // 10),
    report_to=["wandb"], # Set logging to Weights & Biases
    run_name=f"whisper-large-v3-is-lora-bs{PER_DEVICE_BATCH_SIZE}-gpu{NUM_GPUS}", # Optional: Custom run name for wandb
)

# Set W&B Project Name via environment variable (alternative to run_name argument)
# os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

# --- Set Generation Config for Evaluation ---
training_args.generation_config = model.generation_config
training_args.generation_config.language = TARGET_LANGUAGE
training_args.generation_config.task = TASK
#training_args.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=TARGET_LANGUAGE, task=TASK)
training_args.generation_config.forced_decoder_ids = None # <-- Explicitly set to None
training_args.generation_config.suppress_tokens = []


class Bf16PredictionTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`torch.nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            # Default behavior if not generating or only needing loss
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: Modifications for generation handling and explicit autocast
        # Reference:
        # https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/trainer_seq2seq.py#L275

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
             gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
             gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )

        generation_inputs = inputs["input_features"] # Whisper uses input_features

        # ---- START: Explicit Autocast Wrapper ----
        # Ensure generation runs in bf16 context
        autocast_dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else None
        if autocast_dtype:
            with torch.autocast(device_type=inputs["input_features"].device.type, dtype=autocast_dtype):
                generated_tokens = self.model.generate(generation_inputs, **gen_kwargs)
        else:
             # Run without autocast if bf16/fp16 is not enabled (though it is in your case)
             generated_tokens = self.model.generate(generation_inputs, **gen_kwargs)
        # ---- END: Explicit Autocast Wrapper ----


        # Temporary hack to ensure the generation config is correctly picked up from trainer args
        # borrowed from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_seq2seq.py#L275C1-L275C1
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            gen_kwargs.update(self.model.generation_config.to_dict()) # Use instance attribute

        # XXX: adapt synced_gpus for fairscale as well
        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config if hasattr(self.model, "generation_config") else None # Use instance attribute
        # Let's see if we can retrieve the generation config from the model config
        # Maybe we should see if this works better behaviorally?
        # Let's do that after the PR is merged. Follow up?
        # Need to clarify the precedence generation_config > model.config
        if gen_config is None and hasattr(self.model, "config"):
            gen_config = self.model.config # Use instance attribute

        # Fallback to default possible gen_kwargs
        if gen_config is not None:
            gen_kwargs.update(gen_config.to_dict())


        # Rewrite for Seq2Seqlemish models
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        loss = None # Loss is typically not computed during generation step

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

print("Training Arguments configured for wandb.")

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Bf16PredictionTrainer( # Use the custom class
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
#trainer = Seq2SeqTrainer(
#    args=training_args,
#    model=model,
#    train_dataset=dataset["train"],
#    eval_dataset=dataset["test"],
#    data_collator=data_collator,
#    compute_metrics=compute_metrics,
#    tokenizer=processor.feature_extractor,
#)
print("Trainer initialized.")

# --- Start Training ---
print("Starting LoRA fine-tuning on GPU specified by CUDA_VISIBLE_DEVICES...")
# Ensure you've logged into wandb via CLI or set WANDB_API_KEY
train_result = trainer.train()
# To resume: trainer.train(resume_from_checkpoint=True) or specify path

# --- Log Metrics ---
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics) # Saves locally as well
print("Training finished and metrics saved locally and logged to wandb.")

# --- Finish W&B Run ---
if wandb.run is not None:
    wandb.finish()

print("Saving final LoRA adapters and processor...")
# Trainer already saved adapters based on save_strategy and load_best_model_at_end
processor.save_pretrained(training_args.output_dir)

print("Evaluating final model on test set...")
final_metrics = trainer.evaluate(eval_dataset=dataset["test"])
trainer.log_metrics("final_eval", final_metrics) # Logs to wandb if run is still active (it isn't after finish())
trainer.save_metrics("final_eval", final_metrics) # Save final eval metrics locally

print(f"LoRA adapters and processor saved to {training_args.output_dir}")
print("Final Evaluation Metrics:", final_metrics)