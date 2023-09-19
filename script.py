import os

os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_DISABLED"] = "true"

import json
import math
import random
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
import transformers

from .custom_scheduler import FPSchedulerTrainer
from .matplotgraph import create_graph
from .train_utils import get_available_loras_local, precise_cut, sliding_block_cut

from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from peft.utils.other import \
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as model_to_lora_modules
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
)

from modules import shared, utils
from modules.ui import create_refresh_button

from modules.evaluate import (
    calculate_perplexity,
    generate_markdown_table,
    save_past_evaluations
)
from modules.logging_colors import logger
from modules.models import reload_model
from modules.utils import natural_keys


params = {
        "display_name": "Training PRO",
        "is_tab": True
}

non_serialized_params = {
        "debug_slicer": False,
        "Lora_sortedByTime": False,
}

MODEL_CLASSES = {v[1]: v[0] for v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()}
PARAMETERS = ["lora_name", "always_override", "save_steps", "micro_batch_size", "batch_size", "epochs", "learning_rate", "lr_scheduler_type", "lora_rank", "lora_alpha", "lora_dropout", "cutoff_len", "dataset", "eval_dataset", "format", "eval_steps", "raw_text_file", "higher_rank_limit", "warmup_steps", "optimizer", "hard_cut_string", "train_only_after", "stop_at_loss", "add_eos_token", "min_chars", "report_to", "precize_slicing_overlap", "add_eos_token_type", "save_steps_under_loss", "add_bos_token", "training_projection","sliding_window","warmup_ratio"]
WANT_INTERRUPT = False

train_log = {}
train_template = {}
train_log_graph = []
train_choices = ["all","q-k-v-o","q-k-v","k-v-down","q-v"]



def ui():
    with gr.Tab('Train LoRA', elem_id='lora-train-tab'):
        tmp = gr.State('')
        with gr.Row():
            with gr.Column():
                gr.Markdown("`Ver: 23.09.09` This is enhanced version of QLora Training. Maintained by [FP](https://github.com/FartyPants/Training_PRO/tree/main)")

                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            copy_from = gr.Dropdown(label='Copy parameters from', value='None', choices=get_available_loras_local(non_serialized_params['Lora_sortedByTime']), elem_classes=['slim-dropdown'])
                            create_refresh_button(copy_from, lambda: None, lambda: {'choices': get_available_loras_local(non_serialized_params['Lora_sortedByTime'])}, 'refresh-button')
                    with gr.Column():
                        sort_byTime = gr.Checkbox(label='Sort list by Date', value=False, info='Sorts Loras by date created.', elem_classes=['no-background'])                        

                with gr.Row():
                    with gr.Column(scale=5):
                        lora_name = gr.Textbox(label='Name', info='The name of your new LoRA file')
    
                    with gr.Column():
                        always_override = gr.Checkbox(label='Override Existing Files', value=False, info='If the name is the same, checking will replace the existing file, and unchecking will load and continue from it (the rank must be the same).', elem_classes=['no-background'])

                with gr.Row():
                    with gr.Column():
                        lora_rank = gr.Slider(label='LoRA Rank', value=32, minimum=0, maximum=1024, step=4, info='Also called dimension count. Higher values = larger file, more content control. Smaller values = smaller file, less control. Use 4 or 8 for style, 128 or 256 to teach, 1024+ for fine-detail on big data. More VRAM is needed for higher ranks.')
                        lora_alpha = gr.Slider(label='LoRA Alpha', value=64, minimum=0, maximum=2048, step=4, info='This divided by the rank becomes the scaling of the LoRA. Higher means stronger. A good standard value is twice your Rank.')
                        batch_size = gr.Slider(label='Batch Size', value=128, minimum=0, maximum=1024, step=4, info='Global batch size. The two batch sizes together determine gradient accumulation (gradientAccum = batch / microBatch). Higher gradient accum values lead to better quality training.')
                        micro_batch_size = gr.Slider(label='Micro Batch Size', value=4, minimum=1, maximum=128, step=1, info='Per-device batch size (NOTE: multiple devices not yet implemented). Increasing this will increase VRAM usage.')
                        cutoff_len = gr.Slider(label='Cutoff Length', minimum=0, maximum=2048, value=256, step=32, info='Cutoff length for text input. Essentially, how long of a line of text to feed in at a time. Higher values require drastically more VRAM.')

                    with gr.Column():
                        save_steps = gr.Number(label='Save every n steps', value=0, info='If above 0, a checkpoint of the LoRA will be saved every time this many steps pass.')
                        save_steps_under_loss = gr.Slider(label='Save Loss Threshold', value=1.9, minimum=0.0, maximum=3.0, step=0.1, info='Save checkpoints only if the loss is less or equall Threshold loss. (0 = save all)')
                        epochs = gr.Number(label='Epochs', value=3, info='Number of times every entry in the dataset should be fed into training. So 1 means feed each item in once, 5 means feed it in five times, etc.')
                        learning_rate = gr.Textbox(label='Learning Rate', value='3e-4', info='In scientific notation. 3e-4 is a good starting base point. 1e-2 is extremely high, 1e-6 is extremely low.')
                        lr_scheduler_type = gr.Dropdown(label='LR Scheduler', value='linear', choices=['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt', 'FP_low_epoch_annealing', 'FP_half_time_annealing'], info='Learning rate scheduler - defines how the learning rate changes over time. Custom schedulers: FP_low_epoch_annealing - constant for 1 epoch then cosine anneal. FP_half_time_annealing - constant for half time then cosine anneal ', elem_classes=['slim-dropdown'])

                with gr.Accordion(label='Advanced Options', open=True):
                    with gr.Row():
                        with gr.Column():
                            warmup_steps = gr.Number(label='Warmup Steps', value=100, info='Number of max steps used for a linear warmup. Value different than 0 has precedent over Warmup Ratio. The actual number of steps will be the closest multiple of graddient accumulation')
                            warmup_ratio = gr.Slider(label='Warmup Ratio', minimum=0.0, maximum=0.2, step=0.025, value=0.0, info='Ratio of total training steps that will be used for a linear warmup. It applies only if Warmup Step is 0.')
                            stop_at_loss = gr.Slider(label='Stop at loss', minimum=0.0, maximum=3.0, step=0.1, value=0.00, info='The process will automatically stop once the desired loss value is reached. (reasonable numbers are 1.5-1.8)')
                            training_projection = gr.Radio(value = train_choices[4], label='LLaMA Target Projections', info='Change the targets (LORA is typically q-v)', choices=train_choices)    
                            lora_dropout = gr.Slider(label='LoRA Dropout', minimum=0.0, maximum=1.0, step=0.025, value=0.05, info='Percentage probability for dropout of LoRA layers. This can help reduce overfitting. Most users should leave at default.')
                            optimizer = gr.Dropdown(label='Optimizer', value='adamw_torch', choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad'], info='Different optimizer implementation options, for advanced users. Effects of different options are not well documented yet.', elem_classes=['slim-dropdown'])

                        with gr.Column():
                            train_only_after = gr.Textbox(label='Train Only After', value='', info='Only consider text *after* this string in any given chunk for training. For Alpaca datasets, use "### Response:" to only train the response and ignore the input.')
                            add_bos_token = gr.Checkbox(label='Add BOS token', value=True, info="Adds BOS token for each dataset item")
                            add_eos_token = gr.Checkbox(label='Add EOS token', value=False, info="Adds EOS token for each dataset item")
                            add_eos_token_type = gr.Dropdown(label='EOS placement (raw text)', choices=['Every Block', 'Hard Cut Blocks Only'], value='Every Block', info='', allow_custom_value = False)
                            
                            higher_rank_limit = gr.Checkbox(label='Enable higher ranks', value=False, info='If checked, changes Rank/Alpha slider above to go much higher. This will not work without a datacenter-class GPU.')
                            report_to = gr.Radio(label="Save detailed logs with", value="None", choices=["None", "wandb", "tensorboard"], interactive=True)

            with gr.Column():
                with gr.Tab(label='Formatted Dataset'):
                    with gr.Row():
                        format = gr.Dropdown(choices=utils.get_datasets('training/formats', 'json'), value='None', label='Data Format', info='The format file used to decide how to format the dataset input.', elem_classes=['slim-dropdown'])
                        create_refresh_button(format, lambda: None, lambda: {'choices': utils.get_datasets('training/formats', 'json')}, 'refresh-button')

                    with gr.Row():
                        dataset = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'json'), value='None', label='Dataset', info='The dataset file to use for training.', elem_classes=['slim-dropdown'])
                        create_refresh_button(dataset, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'json')}, 'refresh-button')

                    with gr.Row():
                        eval_dataset = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'json'), value='None', label='Evaluation Dataset', info='The (optional) dataset file used to evaluate the model after training.', elem_classes=['slim-dropdown'])
                        create_refresh_button(eval_dataset, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'json')}, 'refresh-button')

                    eval_steps = gr.Number(label='Evaluate every n steps', value=100, info='If an evaluation dataset is given, test it every time this many steps pass.')

                with gr.Tab(label="Raw text file"):
                    with gr.Row():
                        raw_text_file = gr.Dropdown(choices=utils.get_datasets('training/datasets', 'txt'), value='None', label='Text file', info='The raw text file to use for training.', elem_classes=['slim-dropdown'])
                        create_refresh_button(raw_text_file, lambda: None, lambda: {'choices': utils.get_datasets('training/datasets', 'txt')}, 'refresh-button')

                    with gr.Row():
                        with gr.Column():
                            precize_slicing_overlap = gr.Checkbox(label='Add Overlapping blocks', value = True)
                            sliding_window = gr.Checkbox(label='DEMENTROR Learning by FP (Highly Experimental)', value = False, info='Deep Memorization Enforcement Through Overlapping and Repetition. (I named it that, so shush). Special process for learning from long-form text')
                            debug_slicer = gr.Checkbox(label='Dump sentencelist.json to logs/', value = non_serialized_params['debug_slicer'], info='Debug Slicer')

                        with gr.Column():
                            hard_cut_string = gr.Textbox(label='Hard Cut String', value='\\n\\n\\n', info='String that indicates a cut between logical blocks of text (ex. Ideas or Chapters). Helps prevent unwanted overlap between unrelated ideas.')
                            min_chars = gr.Number(label='Ignore small blocks', value=0, info='Ignore Text blocks that have less or equal characters than this number.')

                with gr.Row():
                    start_button = gr.Button("Start LoRA Training", variant='primary')
                    stop_button = gr.Button("Interrupt")

                output = gr.Markdown(value="Ready")

    with gr.Tab('Perplexity evaluation', elem_id='evaluate-tab'):
        with gr.Row():
            with gr.Column():
                models = gr.Dropdown(utils.get_available_models(), label='Models', multiselect=True)
                evaluate_text_file = gr.Dropdown(choices=['wikitext', 'ptb', 'ptb_new'] + utils.get_datasets('training/datasets', 'txt')[1:], value='wikitext', label='Input dataset', info='The raw text file on which the model will be evaluated. The first options are automatically downloaded: wikitext, ptb, and ptb_new. The next options are your local text files under training/datasets.')
                with gr.Row():
                    with gr.Column():
                        stride_length = gr.Slider(label='Stride', minimum=1, maximum=2048, value=512, step=1, info='Used to make the evaluation faster at the cost of accuracy. 1 = slowest but most accurate. 512 is a common value.')

                    with gr.Column():
                        max_length = gr.Slider(label='max_length', minimum=0, maximum=8096, value=0, step=1, info='The context for each evaluation. If set to 0, the maximum context length for the model will be used.')

                with gr.Row():
                    start_current_evaluation = gr.Button("Evaluate loaded model")
                    start_evaluation = gr.Button("Evaluate selected models")
                    stop_evaluation = gr.Button("Interrupt")

            with gr.Column():
                evaluation_log = gr.Markdown(value='')

        evaluation_table = gr.Dataframe(value=generate_markdown_table(), interactive=True)
        with gr.Row():
            save_comments = gr.Button('Save comments', elem_classes="small-button")
            refresh_table = gr.Button('Refresh the table', elem_classes="small-button")

    # Training events
    all_params = [lora_name, always_override, save_steps, micro_batch_size, batch_size, epochs, learning_rate, lr_scheduler_type, lora_rank, lora_alpha, lora_dropout, cutoff_len, dataset, eval_dataset, format, eval_steps, raw_text_file, higher_rank_limit, warmup_steps, optimizer, hard_cut_string, train_only_after, stop_at_loss, add_eos_token, min_chars, report_to, precize_slicing_overlap, add_eos_token_type, save_steps_under_loss, add_bos_token, training_projection,sliding_window,warmup_ratio]

    copy_from.change(do_copy_params, [copy_from] + all_params, all_params)
    start_button.click(do_train, all_params, output)
    stop_button.click(do_interrupt, None, None, queue=False)
    higher_rank_limit.change(change_rank_limit, [higher_rank_limit], [lora_rank, lora_alpha])

    # Evaluation events. For some reason, the interrupt event
    # doesn't work with the .then() syntax, so I write them one
    # by one in this ugly but functional way.
    ev = start_evaluation.click(calculate_perplexity, [models, evaluate_text_file, stride_length, max_length], evaluation_log, show_progress=False)
    start_evaluation.click(generate_markdown_table, None, evaluation_table, show_progress=False)

    start_current_evaluation.click(lambda: ['current model'], None, tmp)
    ev_cur = start_current_evaluation.click(calculate_perplexity, [tmp, evaluate_text_file, stride_length, max_length], evaluation_log, show_progress=False)
    start_current_evaluation.click(generate_markdown_table, None, evaluation_table, show_progress=False)

    stop_evaluation.click(None, None, None, cancels=[ev, ev_cur], queue=False)
    refresh_table.click(generate_markdown_table, None, evaluation_table, show_progress=True)
    save_comments.click(
        save_past_evaluations, evaluation_table, None).then(
        lambda: "Comments saved.", None, evaluation_log, show_progress=False)

    def reload_lora():
        return gr.Dropdown.update(choices=get_available_loras_local(non_serialized_params['Lora_sortedByTime']))
 
    # nonserialized items

    sort_byTime.change(lambda x: non_serialized_params.update({"Lora_sortedByTime": x}), sort_byTime, None).then(reload_lora,None,copy_from) 
    debug_slicer.change(lambda x: non_serialized_params.update({"debug_slicer": x}), debug_slicer, None)


def do_interrupt():
    global WANT_INTERRUPT
    WANT_INTERRUPT = True


def do_copy_params(lora_name: str, *args):
    f_name = f"{shared.args.lora_dir}/{clean_path(None, lora_name)}/training_parameters.json"
    if Path(f_name).is_file():
        with open(f_name, 'r', encoding='utf-8') as format_file:
            params: dict[str, str] = json.load(format_file)
    else:
        params = {}

    result = list()
    for i in range(0, len(PARAMETERS)):
        key = PARAMETERS[i]
        if key in params:
            result.append(params[key])
        else:
            result.append(args[i])

    return result


def change_rank_limit(use_higher_ranks: bool):
    mult = 2 if use_higher_ranks else 1
    return {"maximum": 1024 * mult, "__type__": "update"}, {"maximum": 2048 * mult, "__type__": "update"}


def clean_path(base_path: str, path: str):
    """Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'


def backup_adapter(input_folder):
    # Get the creation date of the file adapter_model.bin
    try:
        adapter_file = Path(f"{input_folder}/adapter_model.bin")
        if adapter_file.is_file():

            logger.info("Backing up existing LoRA adapter...")
            creation_date = datetime.fromtimestamp(adapter_file.stat().st_ctime)
            creation_date_str = creation_date.strftime("Backup-%Y-%m-%d")

            # Create the new subfolder
            subfolder_path = Path(f"{input_folder}/{creation_date_str}")
            subfolder_path.mkdir(parents=True, exist_ok=True)

            # Check if the file already exists in the subfolder
            backup_adapter_file = Path(f"{input_folder}/{creation_date_str}/adapter_model.bin")
            if backup_adapter_file.is_file():
                print(" - Backup already exists. Skipping backup process.")
                return

            # Copy existing files to the new subfolder
            existing_files = Path(input_folder).iterdir()
            for file in existing_files:
                if file.is_file():
                    shutil.copy2(file, subfolder_path)
    except Exception as e:
        print("An error occurred in backup_adapter:", str(e))


def calc_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def do_train(lora_name: str, always_override: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lr_scheduler_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float, cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, raw_text_file: str, higher_rank_limit: bool, warmup_steps: int, optimizer: str, hard_cut_string: str, train_only_after: str, stop_at_loss: float, add_eos_token: bool, min_chars: int, report_to: str, precize_slicing_overlap: bool, add_eos_token_type: str, save_steps_under_loss: float, add_bos_token: bool, training_projection: str,sliding_window:bool,warmup_ratio:float):

    if shared.args.monkey_patch:
        from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
            replace_peft_model_with_int4_lora_model
        )
        replace_peft_model_with_int4_lora_model()

    global WANT_INTERRUPT
    WANT_INTERRUPT = False

    # == Input validation / processing ==
    yield "Preparing the input..."
    lora_file_path = clean_path(None, lora_name)
    if lora_file_path.strip() == '':
        yield "Missing or invalid LoRA file name input."
        return

    lora_file_path = f"{Path(shared.args.lora_dir)}/{lora_file_path}"
    actual_lr = float(learning_rate)
    model_type = type(shared.model).__name__

    if model_type in MODEL_CLASSES:
        model_id = MODEL_CLASSES[model_type]
    else:
        model_id = "llama"
        if model_type == "PeftModelForCausalLM":
            if len(shared.lora_names) > 0:
                yield "You are trying to train a LoRA while you already have another LoRA loaded. This will work, but may have unexpected effects. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
                logger.warning("Training LoRA over top of another LoRA. May have unexpected effects.")
            else:
                yield "Model ID not matched due to LoRA loading. Consider reloading base model. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
                logger.warning("Model ID not matched due to LoRA loading. Consider reloading base model.")
        else:
            yield "LoRA training has only currently been validated for LLaMA, OPT, GPT-J, and GPT-NeoX models. Unexpected errors may follow. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
            logger.warning(f"LoRA training has only currently been validated for LLaMA, OPT, GPT-J, and GPT-NeoX models. (Found model type: {model_type})")

        time.sleep(5)

    if shared.args.loader == 'GPTQ-for-LLaMa' and not shared.args.monkey_patch:
        yield "LoRA training with GPTQ-for-LLaMa requires loading with `--monkey-patch`"
        return

    if cutoff_len <= 0 or micro_batch_size <= 0 or batch_size <= 0 or actual_lr <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        yield "Cannot input zeroes."
        return

    gradient_accumulation_steps = batch_size // micro_batch_size
    shared.tokenizer.pad_token_id = 0
    shared.tokenizer.padding_side = "left"

    def encode(text, prepend_bos_token):
       
        result = shared.tokenizer.encode(text, truncation=True, max_length=cutoff_len)
        # Check if the first two tokens are BOS
        if len(result) >= 2 and result[:2] == [shared.tokenizer.bos_token_id, shared.tokenizer.bos_token_id]:
            result = result[1:]

        if not prepend_bos_token and result[0] == shared.tokenizer.bos_token_id:
            result = result[1:]
        return result

    def tokenize(prompt, append_eos_token=False, prepend_bos_token = False):

        if train_only_after == '' or train_only_after not in prompt:
            input_ids = encode(prompt, prepend_bos_token)

            if append_eos_token and input_ids[-1] != shared.tokenizer.eos_token_id and len(input_ids) < cutoff_len:
                input_ids.append(shared.tokenizer.eos_token_id)

            input_ids = [shared.tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
            
            labels = [1] * len(input_ids)
        else:
            ind = prompt.index(train_only_after) + len(train_only_after)
            before_tokens = encode(prompt[:ind], prepend_bos_token)
            after_tokens = encode(prompt[ind:], False)

            if append_eos_token and after_tokens[-1] != shared.tokenizer.eos_token_id:
                after_tokens.append(shared.tokenizer.eos_token_id)

            full_length = len(after_tokens) + len(before_tokens)
            if full_length > cutoff_len:
                after_tokens = after_tokens[:cutoff_len - len(before_tokens)]
            else:
                before_tokens = [shared.tokenizer.pad_token_id] * (cutoff_len - full_length) + before_tokens

            input_ids = before_tokens + after_tokens
            labels = [-100] * len(before_tokens) + [1] * len(after_tokens)

        input_ids = torch.tensor(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(shared.tokenizer.pad_token_id),
        }

    train_template.clear()

            

    print(f"*** LoRA: {lora_name} ***")

    # END OF FPHAM SENTENCE SPLIT functions ===================     

    # == Prep the dataset, format, etc ==
    if raw_text_file not in ['None', '']:
        train_template["template_type"] = "raw_text"
        logger.info("Loading raw text file dataset...")
        fullpath = clean_path('training/datasets', f'{raw_text_file}')
        fullpath = Path(fullpath)
        if fullpath.is_dir():
            logger.info('Training path directory {}'.format(raw_text_file))
            raw_text = ""
            file_paths = sorted(fullpath.glob('*.txt'), key=lambda path: natural_keys(path.name))
            for file_path in file_paths:
                if file_path.is_file():
                    with file_path.open('r', encoding='utf-8') as file:
                        raw_text += file.read().replace('\r', '')

                    logger.info(f"Loaded training file: {file_path.name}")
        else:
            with open(clean_path('training/datasets', f'{raw_text_file}.txt'), 'r', encoding='utf-8') as file:
                raw_text = file.read().replace('\r', '')
        
        # FPHAM PRECISE SLICING        
        if min_chars<0:
            min_chars = 0

        add_EOS_to_all = add_eos_token and add_eos_token_type == 'Every Block'
        add_EOS_to_HC = add_eos_token and add_eos_token_type != 'Every Block'

        #print (f"add_eos_token {add_eos_token}, add_EOS_to_all {add_EOS_to_all}, add_EOS_to_HC {add_EOS_to_HC}")

        # == New more precise slicing on sentence boundary ==
        if sliding_window:
            text_chunks = sliding_block_cut(raw_text, min_chars, add_EOS_to_HC, cutoff_len, hard_cut_string,non_serialized_params['debug_slicer'])
        else:
            text_chunks = precise_cut(raw_text, precize_slicing_overlap, min_chars, add_EOS_to_HC, cutoff_len, hard_cut_string,non_serialized_params['debug_slicer'])

        train_data = Dataset.from_list([tokenize(x, add_EOS_to_all, add_bos_token) for x in text_chunks])
        if add_EOS_to_all:
            print(f"Added EOS to {len(text_chunks)} blocks") 

        del text_chunks
        eval_data = None
    else:
        if dataset in ['None', '']:
            yield "Missing dataset choice input, cannot continue."
            return

        if format in ['None', '']:
            yield "Missing format choice input, cannot continue."
            return

        train_template["template_type"] = "dataset"

        with open(clean_path('training/formats', f'{format}.json'), 'r', encoding='utf-8-sig') as formatFile:
            format_data: dict[str, str] = json.load(formatFile)

        # == store training prompt ==
        for _, value in format_data.items():
            prompt_key = f"template_{len(train_template)}"
            train_template[prompt_key] = value

        def generate_prompt(data_point: dict[str, str]):
            for options, data in format_data.items():
                if set(options.split(',')) == set(x[0] for x in data_point.items() if (type(x[1]) is str and len(x[1].strip()) > 0)):
                    for key, val in data_point.items():
                        if type(val) is str:
                            data = data.replace(f'%{key}%', val)
                    return data
            raise RuntimeError(f'Data-point "{data_point}" has no keyset match within format "{list(format_data.keys())}"')

        def generate_and_tokenize_prompt(data_point):
            prompt = generate_prompt(data_point)
            return tokenize(prompt, add_eos_token, add_bos_token)

        logger.info("Loading JSON datasets...")
        data = load_dataset("json", data_files=clean_path('training/datasets', f'{dataset}.json'))
        train_data = data['train'].map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))

        print(f"BOS: {add_bos_token} EOS: {add_eos_token}") 

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_data = load_dataset("json", data_files=clean_path('training/datasets', f'{eval_dataset}.json'))
            eval_data = eval_data['train'].map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))

    # == We MUST reload model if it went through any previous training, even failed one ==
    if shared.model_dirty_from_training:
        selected_model = shared.model_name
        if selected_model:
            print("\033[1;31;1m(Model has been modified by previous training, it needs to be reloaded...)\033[0;37;0m")
            try:
                yield f"Reloading {selected_model}..."
                reload_model()
                if shared.model is not None:
                    print("Model reloaded OK, continue with training.")
                else:
                    return f"Failed to load {selected_model}."
            except:
                exc = traceback.format_exc()
                logger.error('Failed to reload the model.')
                print(exc)
                return exc.replace('\n', '\n\n')

    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        logger.info("Getting model ready...")
        prepare_model_for_kbit_training(shared.model)

    # base model is now frozen and should not be reused for any other LoRA training than this one
    shared.model_dirty_from_training = True
    if training_projection==train_choices[0]:
        model_to_lora_modules["llama"] = ["gate_proj","down_proj","up_proj","q_proj","k_proj","v_proj","o_proj"]
    elif training_projection==train_choices[1]:
        model_to_lora_modules["llama"] = ["q_proj","k_proj", "v_proj", "o_proj"]
    elif training_projection==train_choices[2]:
        model_to_lora_modules["llama"] = ["q_proj","k_proj", "v_proj"]
    elif training_projection==train_choices[3]:
        model_to_lora_modules["llama"] = ["k_proj", "v_proj", "down_proj"]        
    else:
        model_to_lora_modules["llama"] = ["q_proj", "v_proj"]            


    logger.info("Preparing for training...")
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=model_to_lora_modules[model_id],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # == Backup the existing adapter ==
    if not always_override:
        backup_adapter(lora_file_path)

    # == get model trainable params
    model_trainable_params, model_all_params = calc_trainable_parameters(shared.model)

    try:
        logger.info("Creating LoRA model...")
        lora_model = get_peft_model(shared.model, config)
        if not always_override and Path(f"{lora_file_path}/adapter_model.bin").is_file():
            logger.info("Loading existing LoRA data...")
            state_dict_peft = torch.load(f"{lora_file_path}/adapter_model.bin")
            set_peft_model_state_dict(lora_model, state_dict_peft)
    except:
        yield traceback.format_exc().replace('\n', '\n\n')
        return

    if shared.args.monkey_patch:
        from alpaca_lora_4bit.autograd_4bit import Autograd4bitQuantLinear
        from alpaca_lora_4bit.models import Linear4bitLt
        for _, m in lora_model.named_modules():
            if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
                if m.is_v1_model:
                    m.zeros = m.zeros.half()
                m.scales = m.scales.half()

    class Tracked():
        def __init__(self):
            self.current_steps = 0
            self.max_steps = 0
            self.did_save = False

    tracked = Tracked()
    actual_save_steps = math.ceil(save_steps / gradient_accumulation_steps)

    class Callbacks(transformers.TrainerCallback):
        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps = state.global_step * gradient_accumulation_steps
            tracked.max_steps = state.max_steps * gradient_accumulation_steps
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True
            elif state.global_step > 0 and actual_save_steps > 0 and state.global_step % actual_save_steps == 0:
                current_loss = float(train_log.get('loss', 0.0))
                if current_loss <= save_steps_under_loss or save_steps_under_loss==0.0:
                    lora_model.save_pretrained(f"{lora_file_path}/checkpoint-{tracked.current_steps}/")
                    print(f"\033[1;30;40mStep: {tracked.current_steps:6} \033[0;37;0m Checkpoint-{tracked.current_steps} saved")
                    # Save log
                    with open(f"{lora_file_path}/checkpoint-{tracked.current_steps}/training_log.json", 'w', encoding='utf-8') as file:
                        json.dump(train_log, file, indent=2)
                    # == Save training prompt ==
                    with open(f"{lora_file_path}/checkpoint-{tracked.current_steps}/training_prompt.json", 'w', encoding='utf-8') as file:
                        json.dump(train_template, file, indent=2)

        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps += 1
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

        def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, logs, **kwargs):
            train_log.update(logs)
            train_log.update({"current_steps": tracked.current_steps})
            if WANT_INTERRUPT:
                print("\033[1;31;1mInterrupted by user\033[0;37;0m")

            print(f"\033[1;30;40mStep: {tracked.current_steps:6} \033[0;37;0m", end='')
            
            entry = {
                'current_steps': int(train_log.get('current_steps',0)),
                'loss': float(train_log.get('loss', 0.0)),
                'learning_rate': float(train_log.get('learning_rate', 0.0)),
                'epoch': float(train_log.get('epoch', 0.0))
            }

            # Add the entry to the continuous log
            train_log_graph.append(entry)

            # Save the graph log for now, we can later generate full graph
            with open(f"{lora_file_path}/training_graph.json", 'w') as file:
                json.dump(train_log_graph, file, indent=4)

            if 'loss' in logs:
                loss = float(logs['loss'])
                if loss <= stop_at_loss:
                    control.should_epoch_stop = True
                    control.should_training_stop = True
                    print(f"\033[1;31;1mStop Loss {stop_at_loss} reached.\033[0;37;0m")

    # FPHAM SAMPLE REQ Transformers error handling
    sample_req = int(train_data.num_rows)//micro_batch_size
    
    if sample_req < gradient_accumulation_steps:
        print(f"\033[1;31;1mWARNING: Current gradient accumulation is too high for the amount of training data.\033[0;37;0m")
        print(f"Gradient accumulation: {gradient_accumulation_steps} should be less than: {sample_req}. \033[1;31;1mThis could crash Accelerate/Transformers\033[0;37;0m")
        min_batchSize = sample_req*micro_batch_size
        print(f"Preferable fix: \033[1;31;1mIncrease the size of dataset\033[0;37;0m")
        print(f"... or Decrerase Batch Size \033[1;31;1m{batch_size}\033[0;37;0m to below {min_batchSize}")
        gradient_accumulation_steps = max(1,sample_req-1)
        print(f"Last resort fix for this run: Lowering Gradient accumulation to {gradient_accumulation_steps}. [Good luck]")

    else:
        print(f"Data Size Check: Gradient accumulation: {gradient_accumulation_steps} <= Data/Batch {sample_req} ... [OK]")

    #END OF FPHAM SAMPLE REQ

    # FPHAM Custom Scheduler ==
    custom_scheduller = False
    lr_scheduler_type_arg = lr_scheduler_type

    if lr_scheduler_type == 'FP_low_epoch_annealing':
        custom_scheduller = True
        lr_scheduler_type_arg = 'cosine'
    elif lr_scheduler_type == 'FP_half_time_annealing':
        custom_scheduller = True
        lr_scheduler_type_arg = 'sine'
    
    args=transformers.TrainingArguments(
            report_to=report_to if report_to != "None" else None,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=math.ceil(warmup_steps / gradient_accumulation_steps),
            warmup_ratio = warmup_ratio,
            num_train_epochs=epochs,
            learning_rate=actual_lr,
            fp16=False if shared.args.cpu else True,
            optim=optimizer,
            logging_steps=1,
            evaluation_strategy="steps" if eval_data is not None else "no",
            eval_steps=math.ceil(eval_steps / gradient_accumulation_steps) if eval_data is not None else None,
            save_strategy="steps" if eval_data is not None else "no",
            output_dir=lora_file_path,
            lr_scheduler_type=lr_scheduler_type_arg,
            load_best_model_at_end=eval_data is not None,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            no_cuda=shared.args.cpu,
        )

    if custom_scheduller:
        trainer = FPSchedulerTrainer(
            model=lora_model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=args,
            data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
            callbacks=list([Callbacks()])
        )
    else:
        trainer = transformers.Trainer(
            model=lora_model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=args,
            data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
            callbacks=list([Callbacks()])
        )
    
    # END OF FPHAM CUSTOM SCHEDULER

    lora_model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        lora_model = torch.compile(lora_model)

    # == Save parameters for reuse ==
    with open(f"{lora_file_path}/training_parameters.json", 'w', encoding='utf-8') as file:
        vars = locals()
        json.dump({x: vars[x] for x in PARAMETERS}, file, indent=2)

    # == Save training prompt ==
    with open(f"{lora_file_path}/training_prompt.json", 'w', encoding='utf-8') as file:
        json.dump(train_template, file, indent=2)

    # == Main run and monitor loop ==
    logger.info("Starting training...")
    yield "Starting..."

    lora_trainable_param, lora_all_param = calc_trainable_parameters(lora_model)

    projections_string = ", ".join([projection.replace("_proj", "") for projection in model_to_lora_modules[model_id]])

    print(f"Training '{model_id}' model using ({projections_string}) projections")

    if lora_all_param > 0:
        print(f"Trainable params: {lora_trainable_param:,d} ({100 * lora_trainable_param / lora_all_param:.4f} %), All params: {lora_all_param:,d} (Model: {model_all_params:,d})")

    train_log.update({"base_model_name": shared.model_name})
    train_log.update({"base_model_class": shared.model.__class__.__name__})
    train_log.update({"base_loaded_in_4bit": getattr(lora_model, "is_loaded_in_4bit", False)})
    train_log.update({"base_loaded_in_8bit": getattr(lora_model, "is_loaded_in_8bit", False)})
    train_log.update({"projections": projections_string})

    if stop_at_loss > 0:
        print(f"Monitoring loss \033[1;31;1m(Auto-Stop at: {stop_at_loss})\033[0;37;0m")

    if WANT_INTERRUPT:
        yield "Interrupted before start."
        return

    def log_train_dataset(trainer):
        decoded_entries = []
        # Try to decode the entries and write the log file
        try:
            # Iterate over the first 10 elements in the dataset (or fewer if there are less than 10)
            for i in range(min(10, len(trainer.train_dataset))):
                decoded_text = shared.tokenizer.decode(trainer.train_dataset[i]['input_ids'])
                decoded_entries.append({"value": decoded_text})

            # Write the log file
            Path('logs').mkdir(exist_ok=True)
            with open(Path('logs/train_dataset_sample.json'), 'w') as json_file:
                json.dump(decoded_entries, json_file, indent=4)

            logger.info("Log file 'train_dataset_sample.json' created in the 'logs' directory.")
        except Exception as e:
            logger.error(f"Failed to create log file due to error: {e}")

    def threaded_run():
        log_train_dataset(trainer)
        trainer.train()
        # Note: save in the thread in case the gradio thread breaks (eg browser closed)
        lora_model.save_pretrained(lora_file_path)
        logger.info("LoRA training run is completed and saved.")
        # Save log
        with open(f"{lora_file_path}/training_log.json", 'w', encoding='utf-8') as file:
            json.dump(train_log, file, indent=2)

    thread = threading.Thread(target=threaded_run)
    thread.start()
    last_step = 0
    start_time = time.perf_counter()

    while thread.is_alive():
        time.sleep(0.5)
        if WANT_INTERRUPT:
            yield "Interrupting, please wait... *(Run will stop after the current training step completes.)*"

        elif tracked.current_steps != last_step:
            last_step = tracked.current_steps
            time_elapsed = time.perf_counter() - start_time
            if time_elapsed <= 0:
                timer_info = ""
                total_time_estimate = 999
            else:
                its = tracked.current_steps / time_elapsed
                if its > 1:
                    timer_info = f"`{its:.2f}` it/s"
                else:
                    timer_info = f"`{1.0/its:.2f}` s/it"

                total_time_estimate = (1.0 / its) * (tracked.max_steps)

            yield f"Running... **{tracked.current_steps}** / **{tracked.max_steps}** ... {timer_info}, {format_time(time_elapsed)} / {format_time(total_time_estimate)} ... {format_time(total_time_estimate - time_elapsed)} remaining"

    # Saving in the train thread might fail if an error occurs, so save here if so.
    if not tracked.did_save:
        logger.info("Training complete, saving...")
        lora_model.save_pretrained(lora_file_path)

    if WANT_INTERRUPT:
        logger.info("Training interrupted.")
        yield f"Interrupted. Incomplete LoRA saved to `{lora_file_path}`."
    else:
        logger.info("Training complete!")
        yield f"Done! LoRA saved to `{lora_file_path}`.\n\nBefore testing your new LoRA, make sure to first reload the model, as it is currently dirty from training."

    create_graph(lora_file_path, lora_name)

def format_time(seconds: float):
    if seconds < 120:
        return f"`{seconds:.0f}` seconds"

    minutes = seconds / 60
    if minutes < 120:
        return f"`{minutes:.0f}` minutes"

    hours = minutes / 60
    return f"`{hours:.0f}` hours"
