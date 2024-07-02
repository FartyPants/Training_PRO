# Training_PRO

This is an expanded and reworked Training tab - the very latest and newest version
Maintained by FP

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q5MOB4M)

Repo home:

https://github.com/FartyPants/Training_PRO

In general the repo is WAY ahead (could be a few months) of the Training PRO extension included in text WebUi. The idea is to keep the extension supplied with the WebUI well tested and stable, while the repo version adds many experimental features that could change shape in further weeks. 

## Installation:

Since a stable version of Training PRO is included in WebUI, to avoid issues with WebUI updates, put this repo in Training_PRO_wip folder and use Training_PRO_wip in Session, instead of the supllied Traing PRO that comes with WebUI

Clone repo to your extensions folder in Training_PRO_wip
```
cd text-generation-webui\extensions\
git clone https://github.com/FartyPants/Training_PRO Training_PRO_wip
```
Now use the Training_PRO_wip in Session, instead of the supllied Traing PRO

![image](https://github.com/FartyPants/Training_PRO/assets/23346289/4778ceff-dd23-4121-ac84-10a0f1c2cd63)

## News

July 2024
- patch for llama 3 padding 
- changed how blocks are eliminated
- shuffled code to make sure the model is reloaded first
- Hybrid training parameters

February 2024
- Hybrid Training (experimental) allows you to use instruct dataset AND Raw text file at the same time creating a hybrid finetune.
  
  ![image](https://github.com/FartyPants/Training_PRO/assets/23346289/856922f8-9bb3-4be0-9d7b-c6727f5df84c)
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
