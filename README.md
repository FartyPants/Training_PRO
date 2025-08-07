# Training_PRO WIP

This is an expanded and reworked Training tab - the very latest and newest version
Maintained by FP

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q5MOB4M)

Repo home:

https://github.com/FartyPants/Training_PRO

In general the repo is WAY ahead (could be a year by now) of the Training PRO extension included in text WebUi. The idea is to keep the extension supplied with the WebUI well tested and stable, while the repo version adds many experimental features that could change shape in further weeks. 


## Training PRO is featured in my huge book "The Cranky Man's Guide to LoRA & QLoRA" 

<img height="200" alt="The Cranky Man's Guide to LoRA & QLoRA" src="https://github.com/user-attachments/assets/afdbaae1-54a6-421f-a52c-ce6ea4477514" />

Find it on [Amazon](https://www.amazon.com/dp/B0FLBTR2FS), [Apple Books](https://books.apple.com/us/book/the-cranky-mans-guide-to-lora-and-qlora/id6749593842), [Kobo](https://www.kobo.com/ca/en/ebook/the-cranky-man-s-guide-to-lora-and-qlora), [Barnes & Noble](https://www.barnesandnoble.com/w/the-cranky-mans-guide-to-lora-and-qlora-f-p-ham/1148001179)


## Installation:

Since a stable version of Training PRO is included in WebUI, to avoid issues with WebUI updates, put this repo in Training_PRO_wip folder and use Training_PRO_wip in Session, instead of the supllied Traing PRO that comes with WebUI

Clone repo to your extensions folder in Training_PRO_wip
```
cd text-generation-webui\extensions\
git clone https://github.com/FartyPants/Training_PRO Training_PRO_wip
```
Now use the Training_PRO_wip in Session, instead of the supllied Traing PRO

![image](https://github.com/FartyPants/Training_PRO/assets/23346289/4778ceff-dd23-4121-ac84-10a0f1c2cd63)

## Blog post
How to convert JSON to JSONL to be used with jinja embedded templates
https://ko-fi.com/post/Convert-JSON-to-JSONL-in-Training-PRO-W7W2100SMQ

## News
May 2025
- changed to the user_data folder WebUI seems to be using now

January 2025
- fix for gradient_checkpoint error when used without BnB
- added new custom schedulers: 
  FP_3Epoch_Raise_Hold_Fall: (min 3 epochs) 1st epoch sine, 2nd epoch Hold, rest of the epochs cosine
  FP_step_decay_with_Warmup - every next epoch will halve the LR
- Coninued Pretraining - adding lm_head and embed_tokens to the training
  ![image](https://github.com/user-attachments/assets/ccf1a12a-2ad4-482a-87d4-0854c4c93a89)
  
  This will do basically sorta full finetune if used with All Linear Targets and the LORA files will be huge (size of the quantized model)
  YOU have to use 8bit optimiser with this, otherwise it won't fit into your 24GB - so you need to use 4-bit BnB to load model, then select 8-bit Adam, like paged_adamw_8bit. You can tune LLAMA-3 8B if you are careful, with rank 16 or 32 and 2-4 batch and 256 context length.
- paged_adamw_8bit and adamw_8bit added
  

July 2024
- patch for llama 3 padding 
- changed how blocks are eliminated
- shuffled code to make sure the model is reloaded first
- Hybrid training parameters
  ![image](https://github.com/FartyPants/Training_PRO/assets/23346289/2054b25f-d4d1-4b23-95dc-c55e738ec6f7)
- tools to convert from JSON to JSONL
  ![image](https://github.com/FartyPants/Training_PRO/assets/23346289/ae61d3ab-5f6b-460c-97c1-ae763e8eca3a)
- dump entire dataset to log

February 2024
- Hybrid Training (experimental) allows you to use instruct dataset AND Raw text file at the same time creating a hybrid finetune.
- Ability to use JSONL (OpenAi) datasets. The format will be chosen automatically from the Template embedded in tokenizer

  ![image](https://github.com/FartyPants/Training_PRO/assets/23346289/81fd0375-3fcb-45a0-a603-c9ad3b8359f9)
- perlexity eval max_length from webui truncation_length_max
- stop at epoch (can be changed during training)
- force bin instead of safetensors (for now)
- remove torch detour and instead set warning ignore
- epoch log is now in 3 decimal numbers instead of 2
- fix for some confusion in raw text over what is EOS token
- Suggestions for Maximum Context length (Measures the longest block in tokens)
- Eliminate cutoff blocks - instead of trimming block if it is above cutoff it will eliminate the block all together. 
- fixes, Group samples by length - makes learning more efficient
- NEFtune: add noise to help with generalization
- Loss Graph in interface.
- Supports Mistral training
- some roundabout around pytorch and transformers version desync

![image](https://github.com/FartyPants/Training_PRO/assets/23346289/e389ec69-d7ad-4922-9ad9-865625997479)

## Features/Changes from Main Training in WebUI

- Chunking: precise raw text slicer (PRTS) uses sentence slicing and making sure things are clean on all ends
- overlap chunking - this special overlapping will make additional overlap block based on logical rules (aka no overlap block on hard cut)
- custom scheduler (follow the code to make your own) In LR Scheduler select FP_low_epoch_annealing - this scheduler will keep the LR constant for first epoch then use cosine for the rest - this part would be best to spawn into a new py file
- saves graph png file at the end with learning rate and loss per epoch
- adding EOS to each block or to hard cut only
- automatically lowers gradient accumulation if you go overboard and set gradient accumulation that will be higher than actual data - transformers would then throw error (or they used to, not sure if still true) but in any way, it will fix bad data
- turn BOS on and OFF
- target selector
- DEMENTOR LEARNING (experimental) Deep Memorization Enforcement Through Overlapping and Repetition. This is an experiment for long-text learning using low epochs (basically use 1 epoch with constant LR or 2 epochs with FP_low_epoch_annealing LR scheduler)
- Getting rid of micro batch size/batch size confusion. Now there is True Batch Size and Gradient accumulation slider, consisten with all the other training out there
- Ability to save Checkpoint during training with a button
- Ability to change Stop Loss during training
- different modes of checkpoint auto saving
- Function to Check Dataset and suggest parameters such as warmup and checkpoint save frequency before training
- Graph Training Loss in interface
- more custom schedulers
  
### Notes:

This uses it's own chunking code for raw text based on sentence splitting. This will avoid weird cuts in the chunks and each chunk should now start with sentence and end on some sentence. It works hand in hand with Hard Cut. A propper use is to structure your text into logical blocks (ideas) separated by three \n then use three \n in hard cut. This way each chunk will contain only one flow of ideas and not derail in the thoughts. And Overlapping code will create overlapped blocks on sentence basis too, but not cross hard cut, thus not cross different ideas either. Does it make any sense? No? Hmmmm...

### Custom schedulers

A bunch of custom (combination) schedulers are added to the LR schedule. These are based on my own experiments

**FP_low_epoch_annealing**

Uses constant LR (with warmup) for 1 epoch only. The rest of the epoch(s) is cosine annealing. So 10 epochs - 1 will be constant 9 will be nose dive down. However a typical usage would be 2 epochs (hence low epoch in name). 1st is constant, the second is annealing. Simple. I use it 90% of time.

**FP_half_time_annealing**

Like the low epoch, but now the total number of steps is divided by 2. First half is constant, second half is annealing. So 10 epochs - 5 will be constant, 5 will be cosine nose down.

**FP_raise_fall_creative**

This is a sine raise till half of the total steps then cosine fall the rest. (Or you may think of the curve as sine in its entirety. The most learning is done in the hump, in the middle. The warmup entry has no effect, since sine is automatically warm up.
The idea is to start very mildly as not to overfit with the first blocks of dataset. It seems to broaden the scope of the model making it less strict for tight dataset. 

### Targets

Normal LORA is q, v and that's what you should use. You can use (q k v o) or (q k v) and it will give you a lot more trainable parameters. The benefit is that you can keep rank lower and still attain the same coherency as q v with high rank. Guanaco has been trained with QLORA and q k v o for example and they swear by it.

### DEMENTOR LEARNING (experimental) Deep Memorization Enforcement Through Overlapping and Repetition

This is and experimental chunking to train long-form text in low number of epochs (basically 1) with sliding repetition. The depth of learning directly depends on the cutoff_length. Increasing cutoff length will also increase number of blocks created from long-form text (which is contrary to normal training). It is based on my own wild experiments. 

### Getting rid of batch size and micro batch size

Keeping consistency with everyone else. 

Listen, There is only ONE batch size - the True batch size (called previously micro-batch size in WebUI) - this is how many blocks are processed at once (during a single step). It eats GPU, but it really helps with the quality training (in fact the ideal batch size would be the same as number of blocks - which is unrealistic) - so the idea is to cram as much True Batch Size before your GPU blows with OOM. On 24GB this is about 10 for 13b (loaded with 4-bit)

So no micro batch size - it is now called True Batch Size, because that's what it is.

The other thing is Gradient Accumulation - this is an emulation of the above Batch size - a virtual batch size, if you will. If your GPU can't handle real batch size then you may fake it using Gradient Accumulation. This will accumulate the gradients over so many steps defined here and then update the weights at the end without increase in GPU.
Gradient accumulation is like a virtual Batch size multiplier without the GPU penalty.

If your batch size is 4 and your gradient accumulation is 2 then it sort of behaves as if we have batch size 8. *Sort of* because Batch size of 4 and GA of 2 is NOT the same as batch size of 2 and GA of 4. (It produces different weights - hence it's not an equivalent). The idea is that if you don't have GPU - using GA to extend batch size is the next best thing (good enough) since you have no other choice.

If all you can afford is 1 batch size, then increasing GA will likely make the learning better in some range of GA (it's not always more is better).

However - GA is not some golden goose. As said, it isn't the same as batch size. In fact GA may worsen your learning as well.

I would suggest a series of experiment where you would put batch size as high as possible without OOM, set GA 1, then repeat training while increasing the GA (2, 4...), and see how the model changes. It's likely that it would follow some sort of curve where GA will seem to help before it will make it worse. Some people believe that if you can squeeze 6 BATCH Size, then you should not bother with GA at all... YMMW

High Batch Size vs High GA would also likely produce different results in terms of learning  words vs style. How? Hmmmm... good question.

One optical "benefit" of GA is that the loss will fluctuate less (because of all the gradient accumulation, which works as a form of noise smoothing as well).

### Eliminating bad blocks

If you use JSON and a block is longer than Maximum context length (Cutoff) by default it will be trimmed to the Maximum context length (Cutoff). That's the default behavior. While it may work on some cases where the end of the block is not much more important than the beginning, in some other cases this may create a really bad situation. Imagine if you are training a text labeling system where you train it on something like this:

USER: Determine the type of text: ... some long text... 

ASSISTANT: Poetry

In such case trimming the block will probably cutoff the entire answer thus making the block useless. Not only that, also skewing the whole functionality where the model may learn that entering long text means no answer will be given.

Elimninate cutoff blocks option will simply not use such block at all. No block is much preferable than having a bad block.

This options apply only to JSON dataset for obvious reasons.

Also watch the terminal window to see how many blocks you had elimitnated - if it is too many, then you need to increase Maximum context length (Cutoff).

### Group Samples by Length

Makes the training more efficient as it will group blocks with simillar size into one batch. The effect can be observed on the loss graph, because the loss will became much more oscillating. This is due to the fact that different block lengths have different loss and when grouped the effect is amplified. 

![image](https://github.com/FartyPants/Training_PRO/assets/23346289/57acfb4c-085a-4d0c-a801-faa11832d413)

### JSONL datasets
Those are the new type of datasets that have role defined. They expect the jinja Template. The format is:

>[
>  {
>    "messages": [
>      {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
>      {"role": "user", "content": "What's the capital of France?"},
>      {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
>    ]
>  },

The format will be chosen autmatically from the chat Template embedded in the tokenizer metadata. If no format is specified (legacy) then the jinja instruction template specified in WebUI will be used

## hybrid training

Did you wonder what would happen if you train partially on raw text and also on instruct dataset? Now you can with Hybrid training. Simply select both files - dataset (and format) and raw text file. And go!
What this will give you - IDK. Experiment. But in general it can stylize instruct response with the writing style of the raw text file. (Of course the correct ratio matters!) Or do some other damage. Now go and test it.



---

### **Training PRO: User Guide**

This Gradio extension provides comprehensive tools for training LoRA (Low-Rank Adaptation) adapters for your LLMs. It covers everything from dataset preparation to advanced training settings and evaluation.

---

### **Overview**

The extension is divided into two main tabs:

1.  **Train LoRA:** Where you configure and start your LoRA training runs.
2.  **Perplexity evaluation:** Where you can evaluate the performance of your models (and LoRAs) on various text datasets.

---

### **1. Train LoRA Tab: Your Fine-Tuning Control Center**

This is where you'll spend most of your time configuring your training.

#### **A. General LoRA Settings**

*   **Ver: 25.05.20:** This just indicates the version of the extension you're using.
*   **Copy parameters from (`copy_from`):**
    *   **How to Use:** Select a previously saved LoRA training run from this dropdown list.
    *   **What it Does:** Fills in *all* the settings on the page with the parameters used for that past LoRA. This is incredibly useful for replicating successful runs or starting new ones based on existing configurations.
*   **Sort list by Date (`sort_byTime`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** Changes the "Copy parameters from" dropdown to sort your saved LoRAs by the date they were created (most recent first) instead of alphabetically.
*   **Name (`lora_name`):**
    *   **How to Use:** Type the name for your new LoRA adapter here. This will be the name of the folder where your LoRA files are saved.
*   **Override Existing Files (`always_override`):**
    *   **How to Use:**
        *   **Checked:** If a LoRA with the same name already exists, checking this box will delete the old one and start a new training from scratch.
        *   **Unchecked:** If a LoRA with the same name exists, the training will *resume* from where it left off, attempting to continue the previous training. (Note: The "LoRA Rank" must be the same as the original for this to work).
    *   **What it Does:** Controls whether you start fresh or continue existing training.

#### **B. LoRA Core Parameters**

These are fundamental settings that define the LoRA adapter itself and the learning process.

*   **LoRA Rank (`lora_rank`):**
    *   **How to Use:** Adjust the slider to set the LoRA rank.
    *   **What it Does:** This is also known as "dimension count." It's the primary factor determining the LoRA's capacity to learn.
        *   **Higher values (e.g., 128, 256, 1024+):** Create a larger LoRA file, allowing it to learn more complex patterns and fine details. Requires more VRAM. Use for teaching new factual information or very specific behaviors.
        *   **Smaller values (e.g., 4, 8, 16):** Create a smaller LoRA file, generally used for stylistic changes, tone, or minor concept adjustments. Requires less VRAM.
*   **LoRA Alpha (`lora_alpha`):**
    *   **How to Use:** Adjust the slider.
    *   **What it Does:** This value scales the LoRA's influence. A common rule of thumb is to set Alpha to 1x or 2x the Rank. The actual scaling is `LoRA_alpha / rank`. Higher alpha means more influence.
*   **Rank Stabilised LoRA (rsLoRA) Accordion:**
    *   **Use rsLoRA (`lora_RS`):**
        *   **How to Use:** Check this box to enable rsLoRA.
        *   **What it Does:** Activates an alternative scaling method where the scale is `rsLoRA_Alpha / sqrt(rank)`. This can sometimes lead to more stable training.
    *   **rsLoRA Alpha (`lora_RS_alpha`):**
        *   **How to Use:** Enter a numerical value.
        *   **What it Does:** Sets the alpha for rsLoRA. A common starting point is 16.
*   **True Batch Size (`micro_batch_size`):**
    *   **How to Use:** Adjust the slider.
    *   **What it Does:** This determines how many "text blocks" (or individual data points) are processed by your GPU in a single training step. Higher values generally lead to more stable training (the model sees more examples at once), but require significantly more VRAM.
*   **Gradient Accumulation Steps (`grad_accumulation`):**
    *   **How to Use:** Adjust the slider.
    *   **What it Does:** This is a clever way to simulate a larger "effective batch size" without requiring more VRAM. If you set `True Batch Size` to 4 and `Gradient Accumulation` to 8, the model will process 4 blocks, store the gradients, then process another 4 blocks, and so on, 8 times. After these 8 steps (32 "virtual" blocks), it updates its weights. This evens out learning but can sometimes reduce the fidelity of training.
*   **Epochs (`epochs`):**
    *   **How to Use:** Enter a number.
    *   **What it Does:** The number of times the entire dataset will be fed through the model for training. If you have 1000 items and set Epochs to 3, the model will see all 1000 items three times. More epochs usually mean more learning but also higher risk of overfitting (where the model memorizes the training data instead of generalizing).
*   **Learning Rate (`learning_rate`):**
    *   **How to Use:** Enter a value in scientific notation (e.g., `3e-4`).
    *   **What it Does:** This controls how much the model's weights are adjusted with each training step.
        *   `3e-4` (0.0003) is a common starting point.
        *   `1e-2` (0.01) is very high, risking unstable training.
        *   `1e-6` (0.000001) is very low, making training slow.
*   **LR Scheduler (`lr_scheduler_type`):**
    *   **How to Use:** Select from the dropdown.
    *   **What it Does:** Defines how the learning rate changes over the course of training.
        *   **`linear`:** Learning rate gradually decreases from its initial value to 0.
        *   **`constant`:** Learning rate stays the same throughout training.
        *   **`constant_with_warmup`:** Stays constant after an initial "warmup" phase.
        *   **`cosine`, `cosine_with_restarts`, `polynomial`, `inverse_sqrt`:** More complex patterns for learning rate decay.
        *   **`FP_low_epoch_annealing` (FP custom):** Starts with a warmup, holds constant for the first epoch, then gradually reduces the learning rate (anneals) for the rest.
        *   **`FP_half_time_annealing` (FP custom):** Warmup, then holds constant for the first half of total steps, then anneals.
        *   **`FP_raise_fall_creative` (FP custom):** Learning rate increases to a peak in the middle of training, then falls.
        *   **`FP_3epoch_raise_hold_fall` (FP custom):** Learning rate "raises" during the 1st epoch, "holds" during the 2nd, and "falls" during the 3rd and subsequent epochs. Designed for at least 3 epochs.
        *   **`FP_step_decay_with_warmup` (FP custom):** Warmup, holds constant for the first epoch, then halves the learning rate with each subsequent epoch.

#### **C. Checkpoints (Saving Your Progress)**

*   **Save every n steps (`save_steps`):**
    *   **How to Use:** Enter a number (0 to disable).
    *   **What it Does:** Your LoRA will be saved as a checkpoint every `n` training steps, and also at the end of each full epoch. This is a basic periodic backup.
*   **Save at 10% Loss change (`save_steps_under_loss`):**
    *   **How to Use:** Adjust the slider to a loss value (e.g., 1.8).
    *   **What it Does:** This is a smart saving feature. Once the training loss falls *below* this value, the system will save a checkpoint. It will then save *again* every time the loss drops by at least 10% from the previous saved checkpoint's loss. This helps capture good states.
*   **Queue Checkpoint Now (`save_chackpoint_now`):**
    *   **How to Use:** Click this button.
    *   **What it Does:** Immediately queues a save operation. The current training step will complete, and then a checkpoint will be saved. Useful for manual backups during a long run.

#### **D. Stops (Automatic Training Termination)**

These settings can be changed *during* training to fine-tune when the process ends.

*   **Stop at loss (`stop_at_loss`):**
    *   **How to Use:** Adjust the slider (0 to disable).
    *   **What it Does:** If the training loss reaches or falls below this specified value, the training will automatically stop. Prevents over-training if you hit a good performance level.
*   **Stop at Epoch (`stop_at_epoch`):**
    *   **How to Use:** Adjust the slider (0 to disable).
    *   **What it Does:** The training will stop at the end of the specified epoch. Useful for limiting training time or comparing specific epoch outputs.

#### **E. Advanced Options (Fine-Grained Control)**

*   **Warmup Steps (`warmup_steps`):**
    *   **How to Use:** Enter a number (0 to use Warmup Ratio instead).
    *   **What it Does:** During these initial steps, the learning rate gradually increases. This "warms up" the model, reducing sudden large updates and preventing early instability.
*   **Warmup Ratio (`warmup_ratio`):**
    *   **How to Use:** Adjust the slider (0.0 to disable, or if Warmup Steps is used).
    *   **What it Does:** If "Warmup Steps" is 0, this ratio determines the portion of total training steps that will be used for a linear warmup. For example, 0.1 means 10% of total steps are for warmup.
*   **NEFtune noise scale (`neft_noise_alpha`):**
    *   **How to Use:** Adjust the slider (0.0 to disable). Recommended starting value: 5.
    *   **What it Does:** Adds a small amount of random noise to the model's embeddings during training. This can help the model generalize better and prevent it from memorizing the training data too strictly.
*   **LLaMA Target Projections (`training_projection`):**
    *   **How to Use:** Select which parts of the model (projections) the LoRA should modify.
    *   **What it Does:** LoRA works by modifying specific internal layers (projections) of the model.
        *   `q-v` (default): Focuses on query and value projections (common for QLoRA).
        *   `all`: Modifies all common projection types.
        *   `q-k-v-o`: Modifies query, key, value, and output projections.
        *   `q-k-v`: Modifies query, key, and value projections.
        *   `k-v-down`: Modifies key, value, and "down" projections.
    *   **Use Case:** Experimenting with these can sometimes yield different results, as it changes which parts of the model's "thinking" are adjusted by the LoRA.
*   **Continued Pretraining Accordion:**
    *   **Train Head (`lora_modulessave`):**
        *   **How to Use:** Check this box.
        *   **What it Does:** Also trains the `lm_head` (the part of the model that predicts the next token) and `embed_tokens` (the part that turns words into numerical representations). This is like "full fine-tuning" for specific parts of the model, which is much more memory intensive.
        *   **Warning:** This requires significantly more VRAM. If enabled, it's *highly recommended* to use an 8-bit AdamW optimizer (like `paged_adamw_8bit`) to manage VRAM.
*   **Use Gradient Checkpoint (`use_grad_checkpoint`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** Reduces VRAM usage during training but makes the training process slower. It achieves this by re-calculating some intermediate values during the backward pass instead of storing them.
*   **LoRA Dropout (`lora_dropout`):**
    *   **How to Use:** Adjust the slider (e.g., 0.05 for 5%).
    *   **What it Does:** Introduces a small chance that some LoRA layers will be temporarily "dropped out" during training. This can prevent overfitting by forcing the model to learn more robust features. Most users can leave this at default.
*   **Optimizer (`optimizer`):**
    *   **How to Use:** Select from the dropdown.
    *   **What it Does:** The algorithm used to update the model's weights during training. `adamw_torch` is a good general choice. `paged_adamw_8bit` is excellent for saving VRAM, especially when "Train Head" is enabled. Different optimizers can affect training speed and final quality.
*   **Train Only After (`train_only_after`):**
    *   **How to Use:** Type a specific string (e.g., `### Response:`)
    *   **What it Does:** If you're using instruction-response datasets (like Alpaca), you often only want the model to learn from the "response" part, not the "instruction" part. This setting tells the model to ignore the loss calculation for any text *before* this string in your data, focusing only on the part after it.
*   **Train on Inputs (Full sequence loss) (`train_on_inputs`):**
    *   **How to Use:** Check or uncheck this box.
    *   **What it Does:**
        *   **Checked (default):** The model calculates loss and trains on the *entire* sequence of text (the input/instruction *plus* the response).
        *   **Unchecked (recommended for instruction tuning with `Train Only After`):** If `Train Only After` is specified, the model *only* calculates loss and trains on the portion of text *after* that string. This is ideal for teaching a model to generate good responses to prompts without altering its understanding of the prompts themselves.
*   **Add BOS token (`add_bos_token`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** Adds a "Beginning Of Sequence" token (e.g., `<s>`) to the start of each text item. This helps the model understand that a new sequence is beginning. Generally, leave this ON.
*   **Add EOS token (`add_eos_token`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** Adds an "End Of Sequence" token (e.g., `</s>`) to the end of each text item. This helps the model understand when to stop generating text. For JSONL datasets, this is typically controlled by the chat template.
*   **EOS placement (Text file) (`add_eos_token_type`):**
    *   **How to Use:** Select from the dropdown.
    *   **What it Does:**
        *   **`Every Block`:** Adds an EOS token at the end of every processed text block.
        *   **`Hard Cut Blocks Only`:** Only adds an EOS token at the end of text blocks that were explicitly separated by the "Hard Cut String."
*   **Group Samples by Length (`group_by_length`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** Groups training examples of similar lengths together. This can make training more efficient by reducing the amount of padding needed for shorter sequences within a batch.
*   **Eliminate cutoff blocks (`eliminate_long_blocks`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** If a text block (or a JSON entry after formatting) exceeds the "Maximum context length (Cutoff)" even after trimming, this option will *remove* that block entirely from the dataset instead of just truncating it. Useful for ensuring all training data fits perfectly.
*   **Enable higher ranks (`higher_rank_limit`):**
    *   **How to Use:** Check this box.
    *   **What it Does:** Increases the maximum values available on the "LoRA Rank" and "LoRA Alpha" sliders.
    *   **Warning:** Only use this if you have a datacenter-class GPU with extremely high VRAM (e.g., 80GB+) as higher ranks consume much more memory.
*   **Save detailed logs with (`report_to`):**
    *   **How to Use:** Select `wandb` (Weights & Biases) or `tensorboard` to integrate with these logging platforms.
    *   **What it Does:** If selected, detailed training metrics, losses, and other information will be sent to the chosen platform for more advanced visualization and tracking.

#### **F. Dataset & Data Preparation Tabs**

This section is crucial for feeding your model the right data. You must select one primary data source (JSON, JSONL, or Text File).

*   **JSON Dataset Tab:** For structured JSON files (list of dictionaries, e.g., Alpaca format).
    *   **Dataset (`dataset`):**
        *   **How to Use:** Select your JSON training file (e.g., `alpaca_data.json`) from `user_data/training/datasets/`.
    *   **Evaluation Dataset (`eval_dataset`):**
        *   **How to Use:** Select an optional separate JSON file for evaluation.
        *   **What it Does:** The model will periodically be tested on this data to monitor its progress and prevent overfitting.
    *   **Data Format (`format`):**
        *   **How to Use:** Select a JSON format file (e.g., `alpaca.json`) from `user_data/training/formats/`.
        *   **What it Does:** This file tells the trainer how to combine different fields (like `instruction`, `input`, `output`) from your JSON dataset into a single text sequence that the model can understand.
    *   **Evaluate every n steps (`eval_steps`):**
        *   **How to Use:** Enter a number.
        *   **What it Does:** If an evaluation dataset is provided, the model's performance will be checked every `n` training steps.

*   **JSONL Dataset Tab:** For JSON Lines files, typically used with chat-based models.
    *   **JSONL Dataset (`datasetJSONL`):**
        *   **How to Use:** Select your JSONL training file (e.g., `chat_data.jsonl`) from `user_data/training/datasets/`.
    *   **JSONL Evaluation Dataset (`eval_datasetJSONL`):**
        *   **How to Use:** Select an optional JSONL evaluation file.
    *   **Evaluate every n steps (`eval_stepsJSONL`):**
        *   **How to Use:** Enter a number.
        *   **What it Does:** Evaluation frequency for JSONL datasets.
    *   **Note:** The format for JSONL files is automatically derived from the model's tokenizer chat template. If your model's tokenizer doesn't have one, it will fall back to the "Instruction template" set in `text-generation-webui`'s "Parameters" tab.
    *   **Important:** This JSONL processing automatically appends `<|EOFUSER|>` to user messages, which is then removed after applying the chat template.

*   **Text file Tab:** For raw `.txt` files where the entire text is used for training.
    *   **Text file (`raw_text_file`):**
        *   **How to Use:** Select your raw text file (e.g., `my_novel.txt`) or a folder containing multiple `.txt` files from `user_data/training/datasets/`.
    *   **Add Overlapping blocks (`precize_slicing_overlap`):**
        *   **How to Use:** Check this box.
        *   **What it Does:** When splitting your raw text into manageable blocks, this creates additional "overlapping" blocks. For example, if block 1 is sentences A-B-C, an overlapping block might be sentences B-C-D. This helps the model learn context across block boundaries.
    *   **DEMENTOR Long-form Learning by FP (Highly Experimental, use low epochs) (`sliding_window`):**
        *   **How to Use:** Check this box.
        *   **What it Does:** Activates a special "Deep Memorization Enforcement Through Overlapping and Repetition" slicing strategy. It uses a "sliding window" approach to generate text blocks. This is intended for teaching the model very long-form text patterns, potentially with fewer epochs. It's experimental, so use with caution.
    *   **Hard Cut String (`hard_cut_string`):**
        *   **How to Use:** Enter a string (e.g., `\\n\\n\\n` for three newlines).
        *   **What it Does:** This string indicates a definitive logical break in your text (like a chapter break or a new topic). The slicer will ensure that no text blocks or overlapping blocks cross this string, preserving logical boundaries.
    *   **Ignore small blocks (`min_chars`):**
        *   **How to Use:** Enter a number.
        *   **What it Does:** Any text block generated during slicing that has fewer characters than this number will be discarded from the training dataset. Useful for removing noise or very short, uninformative segments.

*   **Hybrid Tab (Experimental):** For training with both structured (JSON/JSONL) and unstructured (raw text) data simultaneously.
    *   **Hybrid Training (`hybrid_training`):**
        *   **How to Use:** Check this box.
        *   **What it Does:** Enables the hybrid training mode. You must also select a `Raw Text file` *and* a `JSON` or `JSONL` dataset.
    *   **Percentage of Dataset used (`hybrid_data_ratio`):**
        *   **How to Use:** Adjust the slider (0-100%).
        *   **What it Does:** Controls how much of the selected JSON or JSONL dataset will be included in the hybrid training.
    *   **Percentage of Text file used (`hybrid_text_ratio`):**
        *   **How to Use:** Adjust the slider (0-100%).
        *   **What it Does:** Controls how much of the selected raw text file will be included in the hybrid training.
    *   **Use Case:** This is experimental but can be powerful for blending instruction-following (from JSON/JSONL) with general knowledge or style (from raw text).

*   **URL Tab:** For downloading datasets directly from a URL.
    *   **Download JSON or txt file to datasets (or formats) folder (`download_file_url`):**
        *   **How to Use:** Paste the URL of your `.json` or `.txt` file. For GitHub, use the "Raw" URL. For Hugging Face, ensure the URL contains `/resolve/` not `/blob/`.
    *   **Overwrite (`download_check_overwrite`):**
        *   **How to Use:** Check this if you want to replace an existing file with the same name.
    *   **Destination (`download_folder`):**
        *   **How to Use:** Select whether to save the downloaded file to the `datasets` folder or the `formats` folder.
    *   **Download (`download_button`):**
        *   **How to Use:** Click to start the download.
    *   **Download Status (`download_status`):** Displays progress and messages.

*   **Tools Tab:** Utility functions for preparing your datasets.
    *   **Evaluation dataset split (percentage) (`split_dataset_perc`) & Split dataset (`split_dataset_do`):**
        *   **How to Use:** Select a JSON dataset in the "JSON Dataset" tab, set the percentage (e.g., 10 for a 90/10 split), then click "Split dataset."
        *   **What it Does:** Takes your selected JSON dataset and splits it into two new files: `your_dataset_name_train.json` and `your_dataset_name_eval.json`, according to the specified percentage. It then automatically selects these new files for your training and evaluation datasets.
    *   **Convert JSON to JSONL (`convert_system`) & Convert JSON to JSONL (`convert_do`):**
        *   **How to Use:** Select a JSON dataset, add a "System Message" (if desired), then click "Convert JSON to JSONL."
        *   **What it Does:** Converts your standard JSON dataset (with `instruction` and `output` fields) into the JSONL format, suitable for chat models. The "System Message" will be added to each entry.
    *   **Simple TXT to JSONL conversion (`convert_system2`, `convert_prompt2`) & Convert TXT to JSONL (`convert_do2`):**
        *   **How to Use:** Select a raw text file (where each logical item is separated by at least three empty lines), add a "System Message," add a "Prompt" that will be applied to every item (e.g., "Write me a limerick."), then click "Convert TXT to JSONL."
        *   **What it Does:** Converts your raw text file into a JSONL format where each block of text becomes an assistant's response to your specified prompt and system message.
    *   **Dump Training Dataset (`dump_dataset`) & Clean up dump dataset (`dump_dataset_remove_s`):**
        *   **How to Use:** Check "Dump Training Dataset" to enable. Optionally check "Clean up dump dataset" to remove BOS/EOS tokens.
        *   **What it Does:** Just before training begins, the tool will decode your entire prepared training dataset (after all formatting and slicing) and save it as a JSON file in the `logs/` folder. This is invaluable for debugging and verifying how your data is actually being fed to the model.

#### **G. Final Configuration and Execution**

*   **Maximum context length (Cutoff) (`cutoff_len`):**
    *   **How to Use:** Adjust the slider.
    *   **What it Does:** This is the *maximum number of tokens* that any single block of text (from JSON, JSONL, or raw text) will have. If a block is longer, it will be truncated. Higher values require significantly more VRAM and can slow down training. This is a critical setting for managing GPU memory.
*   **Verify Dataset/Text File and suggest data entries (`check_dataset_btn`):**
    *   **How to Use:** Click this button *before* starting training.
    *   **What it Does:** Analyzes your selected dataset(s) and current settings. It provides a summary in the "Dataset info" textbox, including:
        *   Number of blocks.
        *   Length of the longest block (in tokens), and suggestions for optimal `Cutoff` length.
        *   Total number of tokens.
        *   Calculated total training steps, steps per epoch.
        *   Suggestions for `Save every n steps` and `Warmup steps`.
        *   **Crucial Warning:** It will tell you if your "Gradient Accumulation" setting is too high for your dataset size, which can cause Accelerate/Transformers to crash.
    *   **Recommendation:** Always run this before a new training session!
*   **Start LoRA Training (`start_button`):**
    *   **How to Use:** After configuring everything, click this to begin.
    *   **What it Does:** Initializes the training process. The "Output" area will show real-time status updates, and the "Graph" will plot loss.
*   **Interrupt (`stop_button`):**
    *   **How to Use:** Click this button at any time during training.
    *   **What it Does:** Politely tells the training process to stop after the current training step is finished. It will then save the current state of your LoRA.
*   **Graph (`plot_graph`):**
    *   **What it Does:** Displays a live plot of your training loss over epochs/steps. This is an essential visual aid for monitoring training progress and detecting issues like overfitting (loss goes up) or non-convergence.
*   **Output (`output`):**
    *   **What it Does:** Shows textual status updates, progress indicators (steps, time), and final messages (e.g., "Done! LoRA saved to...").

---

### **2. Perplexity Evaluation Tab: Assessing Your Model's Performance**

This tab allows you to measure how well your loaded model (or selected other models) can predict text in various datasets. Lower perplexity generally means a better model.

*   **Models (`models`):**
    *   **How to Use:** Select one or more base models (and their loaded LoRAs) from your `text-generation-webui` model directory.
    *   **What it Does:** These are the models that will be evaluated.
*   **Input dataset (`evaluate_text_file`):**
    *   **How to Use:** Select a text file for evaluation. Options include:
        *   `wikitext`, `ptb`, `ptb_new`: Standard public datasets (automatically downloaded).
        *   Your local `.txt` files from `user_data/training/datasets/`.
*   **Stride (`stride_length`):**
    *   **How to Use:** Adjust the slider (e.g., 512).
    *   **What it Does:** This setting balances evaluation speed and accuracy. A stride of 1 means the model evaluates every single token, which is very slow but most accurate. A higher stride (e.g., 512) means the model "jumps" ahead by that many tokens for each new evaluation, making it much faster but less precise.
*   **max_length (`max_length`):**
    *   **How to Use:** Enter a number (0 to use the model's full context length).
    *   **What it Does:** Sets the maximum context (number of tokens) the model will use for each evaluation chunk. If 0, it uses the maximum context length your model was trained for.
*   **Evaluate loaded model (`start_current_evaluation`):**
    *   **How to Use:** Click this button.
    *   **What it Does:** Evaluates the *currently loaded* model in your `text-generation-webui` session on the selected input dataset.
*   **Evaluate selected models (`start_evaluation`):**
    *   **How to Use:** Select models from the "Models" dropdown, then click this button.
    *   **What it Does:** Evaluates all the models you've selected from the dropdown on the input dataset.
*   **Interrupt (`stop_evaluation`):**
    *   **How to Use:** Click this button during an evaluation.
    *   **What it Does:** Stops the evaluation process.
*   **Evaluation Log (`evaluation_log`):**
    *   **What it Does:** Displays real-time messages and results of the current evaluation.
*   **Evaluation Table (`evaluation_table`):**
    *   **What it Does:** Shows a historical table of all your past perplexity evaluations, including model names, datasets, perplexity scores, and any comments you've added. It's interactive, so you can edit comments directly.
*   **Save comments (`save_comments`):**
    *   **How to Use:** Click this after editing comments in the "Evaluation Table."
    *   **What it Does:** Saves any changes you've made to the comments column in the table.
*   **Refresh the table (`refresh_table`):**
    *   **How to Use:** Click this button.
    *   **What it Does:** Updates the "Evaluation Table" to ensure it reflects the latest evaluation results and saved comments.

---

### **General Tips and Warnings for Training PRO:**

*   **Model Reload After Training:** After any training run (even if interrupted or failed), your base model becomes "dirty" in memory. You *must* go to the `text-generation-webui`'s "Model" tab and **reload your base model** before using it for inference or starting another training run.
*   **VRAM Management:** LoRA training, especially with higher ranks or `Train Head` enabled, can be very VRAM-intensive. Monitor your GPU usage. Reduce `micro_batch_size`, increase `Gradient Accumulation Steps`, decrease `cutoff_len`, or lower `LoRA Rank` if you hit VRAM limits.
*   **Disk Space:** LoRA checkpoints can accumulate quickly, and a full LoRA can be several GBs. Ensure you have enough disk space.
*   **Tokenizer Issues:** The tool attempts to handle common tokenizer problems (like missing PAD or EOS tokens), but a good, well-behaved tokenizer from your base model is always best. Pay attention to warnings in the console about tokenizers.
*   **Experimentation:** LoRA training often involves experimentation. Start with smaller datasets or fewer epochs to test your settings before committing to long, resource-intensive runs.
*   **The "Instruct" Tab / Console:** Keep an eye on the command line console where `text-generation-webui` is running. Training PRO outputs a lot of useful information, warnings, and debug messages there.

This guide should give you a solid foundation for navigating and utilizing the powerful features of the "Training PRO" extension!
