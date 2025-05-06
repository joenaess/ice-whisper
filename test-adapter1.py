import torch
import librosa
from datasets import Audio # Or use librosa directly
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from peft import PeftModel, PeftConfig
import os
import gc # For garbage collection

# --- Configuration ---
MODEL_ID = "openai/whisper-large-v3"
# !! MODIFY THESE PATHS !!
ADAPTER_CHECKPOINT_PATH = "./whisper-large-v3-is-raddromur-lora-wandb/checkpoint-180"
AUDIO_FILE_PATH = "./RADDROMUR_22.09/speech/nedanmals/nedanmals_000001/nedanmals_000001-0001-00:00:1810-00:00:1662.flac"
#AUDIO_FILE_PATH = "./jonas-icelandic.flac" # Path to your test audio file
# !! ------------- !!

TARGET_LANGUAGE = "is"
TASK = "transcribe"
MODEL_PRECISION = torch.bfloat16 # Use the same precision as training (bfloat16 for A100)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Using model precision: {MODEL_PRECISION}")

# --- 1. Load Processor ---
print(f"\nLoading processor for {MODEL_ID}...")
try:
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=TARGET_LANGUAGE, task=TASK)
    print("Processor loaded.")
except Exception as e:
    print(f"Error loading processor: {e}")
    exit()

# --- 2. Load and Prepare Audio ---
print(f"\nLoading and preparing audio file: {AUDIO_FILE_PATH}...")
try:
    # Option A: Using datasets.Audio (handles resampling)
    # audio_dataset = Dataset.from_dict({"audio": [AUDIO_FILE_PATH]}).cast_column("audio", Audio(sampling_rate=16000))
    # speech_array = audio_dataset[0]["audio"]["array"]
    # sampling_rate = audio_dataset[0]["audio"]["sampling_rate"]

    # Option B: Using librosa (more direct)
    speech_array, sampling_rate = librosa.load(AUDIO_FILE_PATH, sr=16000) # Resample to 16kHz directly

    print(f"Audio loaded. Original SR: {sampling_rate} (resampled to 16k), Duration: {len(speech_array)/sampling_rate:.2f}s")

    # Prepare input features
    input_features = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(DEVICE, dtype=MODEL_PRECISION) # Move to device and set precision
    print("Input features prepared and moved to device.")

except Exception as e:
    print(f"Error loading or processing audio: {e}")
    exit()

# --- 3. Inference with Base Model ---
print(f"\n--- Running Inference with Base Model ({MODEL_ID}) ---")
base_transcription = "Error during base model inference."
try:
    print("Loading base model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_PRECISION,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" # Use the same attn implementation as training
    ).to(DEVICE)
    base_model.eval() # Set to evaluation mode
    print("Base model loaded.")

    # Set generation config (important!)
    base_model.generation_config.language = TARGET_LANGUAGE
    base_model.generation_config.task = TASK
    base_model.generation_config.forced_decoder_ids = None # Ensure no conflicting forced IDs
    base_model.generation_config.suppress_tokens = []


    print("Generating transcription (base model)...")
    with torch.inference_mode(): # Disable gradient calculation for inference
         with torch.autocast(device_type=DEVICE.type, dtype=MODEL_PRECISION): # Use autocast for consistency if needed
            predicted_ids_base = base_model.generate(input_features, generation_config=base_model.generation_config)

    print("Decoding transcription (base model)...")
    base_transcription = processor.batch_decode(predicted_ids_base, skip_special_tokens=True)[0]
    print("\n>>> Base Model Transcription:")
    print(base_transcription)

except Exception as e:
    print(f"Error during base model inference: {e}")
finally:
    # --- Crucial: Clean up VRAM ---
    print("\nUnloading base model from GPU memory...")
    if 'base_model' in locals():
        del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")


# --- 4. Inference with Fine-tuned Adapter ---
print(f"\n--- Running Inference with Fine-tuned Adapter ({ADAPTER_CHECKPOINT_PATH}) ---")
adapter_transcription = "Error during adapter model inference."
try:
    print("Loading base model again...")
    # Need to reload the base model before applying adapter
    base_model_for_adapter = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_PRECISION,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" # Use the same attn implementation
    ) # Load to CPU first to save VRAM if adapter loading is heavy

    print("Loading LoRA adapter...")
    # Load the PeftModel - this merges the adapter onto the base model
    peft_model = PeftModel.from_pretrained(base_model_for_adapter, ADAPTER_CHECKPOINT_PATH)
    peft_model = peft_model.to(DEVICE) # Move the combined model to GPU
    peft_model.eval() # Set to evaluation mode
    print("Adapter loaded and merged model moved to device.")

    # Set generation config on the Peft model (might inherit, but explicit is safer)
    # Note: Access generation_config through the base_model if needed, PEFT forwards attributes
    peft_model.generation_config.language = TARGET_LANGUAGE
    peft_model.generation_config.task = TASK
    peft_model.generation_config.forced_decoder_ids = None
    peft_model.generation_config.suppress_tokens = []


    print("Generating transcription (adapter model)...")
    with torch.inference_mode(): # Disable gradient calculation for inference
         with torch.autocast(device_type=DEVICE.type, dtype=MODEL_PRECISION):
            # Use the same input_features as before
            predicted_ids_adapter = peft_model.generate(input_features, generation_config=peft_model.generation_config)

    print("Decoding transcription (adapter model)...")
    adapter_transcription = processor.batch_decode(predicted_ids_adapter, skip_special_tokens=True)[0]
    print("\n>>> Adapter Model Transcription:")
    print(adapter_transcription)

except Exception as e:
    print(f"Error during adapter model inference: {e}")
finally:
    # --- Clean up VRAM ---
    print("\nUnloading adapter model from GPU memory...")
    if 'peft_model' in locals():
        del peft_model
    if 'base_model_for_adapter' in locals():
        del base_model_for_adapter
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

# --- 5. Comparison ---
print("\n--- Comparison ---")
print(f"Audio File: {AUDIO_FILE_PATH}")
print("-" * 20)
print("Base Model Output:")
print(base_transcription)
print("-" * 20)
print("Adapter Model Output:")
print(adapter_transcription)
print("-" * 20)