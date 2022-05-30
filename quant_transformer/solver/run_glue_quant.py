# TODO: compare with mqbench for quantized_module and get the same result
import os
import numpy as np
import logging
import sys
import argparse
import transformers
from torch.nn import MSELoss
from scipy.optimize import curve_fit
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
)
import datasets
import random
from datasets import load_metric
from transformers.trainer_utils import get_last_checkpoint
import torch  # noqa E401
import copy
import glue_utils
from quant_transformer.model.quant_bert import QuantizedBertForSequenceClassification
from quant_transformer.model.quant_bert_LN import QuantizedBertForSequenceClassification as QuantizedBertForSequenceClassificationLN
from quant_transformer.model.quant_bart import QuantizedBartForSequenceClassification
from quant_transformer.model.quant_roberta import QuantizedRobertaForSequenceClassification
from quant_transformer.model.quant_roberta_LN import QuantizedRobertaForSequenceClassification as QuantizedRobertaForSequenceClassificationLN
from quant_transformer.quantization import enable_calibration_woquantization, enable_quantization, \
    enable_cosine_function, disable_all,  enable_calibration_quantization
from quant_transformer.quantization.observer import ObserverBase  # noqa E401
from quant_transformer.quantization.fake_quant import LSQPlusFakeQuantize, QuantizeBase
from analysis import analysis_quantize
logger = logging.getLogger("transformer")


def set_logger(config_progress):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config_progress.log_level
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def make_huggingface_training_args(config_train, config_progress):
    training_args = TrainingArguments(
        seed=config_train.seed,
        output_dir=config_train.output_dir,
        overwrite_output_dir=config_train.overwrite_output_dir,
        do_train=config_train.do_train,
        do_eval=config_train.do_eval,
        do_predict=config_train.do_predict,
        evaluation_strategy=config_train.evaluation_strategy,
        eval_steps=config_train.eval_steps,
        per_device_train_batch_size=config_train.per_device_train_batch_size,
        per_device_eval_batch_size=config_train.per_device_eval_batch_size,
        gradient_accumulation_steps=config_train.gradient_accumulation_steps,
        learning_rate=config_train.learning_rate,
        weight_decay=config_train.weight_decay,
        max_grad_norm=config_train.max_grad_norm,
        num_train_epochs=config_train.num_train_epochs,
        max_steps=config_train.max_steps,
        lr_scheduler_type=config_train.lr_scheduler_type,
        warmup_ratio=config_train.warmup_ratio,
        warmup_steps=config_train.warmup_steps,
        gradient_checkpointing=config_train.gradient_checkpointing,
        log_level=config_progress.log_level,
        log_level_replica=config_progress.log_level_replica,
        logging_dir=config_progress.logging_dir,
        logging_strategy=config_progress.logging_strategy,
        logging_steps=config_progress.logging_steps,
        save_strategy=config_progress.save_strategy,
        save_steps=config_progress.save_steps,
        save_total_limit=config_progress.save_total_limit,
        save_on_each_node=config_progress.save_on_each_node,
        no_cuda=config_progress.no_cuda,
        run_name=config_progress.run_name,
        disable_tqdm=config_progress.disable_tqdm,
        load_best_model_at_end=config_progress.load_best_model_at_end,
        metric_for_best_model=config_progress.metric_for_best_model,
        greater_is_better=config_progress.greater_is_better
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config_progress.log_level = training_args.get_process_log_level()
    return training_args


def train_from_last_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def train(trainer, training_args):
    checkpoint = None
    last_checkpoint = train_from_last_checkpoint(training_args)
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def evaluate(trainer, eval_datasets, num_samples=-1):
    logger.info("*** Evaluate ***")
    if isinstance(eval_datasets, tuple):
        for i in range(len(eval_datasets)):
            if num_samples != -1:
                metrics = trainer.evaluate(eval_dataset=eval_datasets[i].shuffle().select(range(num_samples)))
            else:
                metrics = trainer.evaluate(eval_dataset=eval_datasets[i])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
    else:
        if num_samples != -1:
            metrics = trainer.evaluate(eval_dataset=eval_datasets.shuffle().select(range(num_samples)))
        else:
            metrics = trainer.evaluate(eval_dataset=eval_datasets)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def get_cali_data(model, data_loader):
    with torch.no_grad():
        cali_data, fp32_output = [], []
        for p in data_loader:
            tmp = {}
            for k, v in p.items():
                tmp[k] = v.cuda()
            del tmp['labels']
            output = model(**tmp)[0]
            cali_data.append(tmp)
            fp32_output.append(output)
    return cali_data, fp32_output


def enable_act_calibration(model, percentile, do_train):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if 'act' in name:
                module.observer.set_percentile(percentile)
                module.observer.cnt = 0
                module.disable_fake_quant()
                module.enable_observer()
            if not do_train and 'weight' in name:
                module.disable_fake_quant()


def enable_act_quant(model, do_train):
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if 'act' in name:
                module.enable_fake_quant()
                module.disable_observer()
            if not do_train and 'weight' in name:
                module.enable_fake_quant()


def find_percentile(model, cali_data, fp32_output, iters, do_train):
    p, loss = 0, 10000
    loss_list = []
    for i in range(iters):
        enable_act_calibration(model, 1.0 - 0.01 * i, do_train)
        calibration(model, cali_data)
        enable_act_quant(model, do_train)
        cur_loss = calibration(model, cali_data, fp32_output)
        print('the iters is {}, the loss is {}'.format(i, cur_loss.item()))
        loss_list.append(cur_loss)
    for i in range(iters):
        cur_loss = loss_list[i]
        if loss > cur_loss:
            loss = cur_loss
            p = i
    return 1.0 - 0.01 * p


loss_fct = MSELoss()


def calibration(model, cali_data, fp_output=None):
    loss = 0
    with torch.no_grad():
        for i, batch in enumerate(cali_data):
            output = model(**batch)[0]
            if fp_output is not None:
                loss += loss_fct(output, fp_output[i])
    return loss


def evaluate_token(model, cali_data, fp32_output, bit, do_train):
    logger.info("*** Evaluate Token Percentile ***")
    if bit == 8:
        iters = 10
    elif bit == 6:
        iters = 30
    else:
        iters = 90
    percentile = find_percentile(model, cali_data, fp32_output, iters, do_train)
    print('the best percentile is {}'.format(percentile))
    enable_act_calibration(model, percentile, do_train)
    calibration(model, cali_data)


def delay_roberta_ln(fp32_model, model):
    for i in range(12):
        fp32 = fp32_model.roberta.encoder.layer[i]
        quant = model.roberta.encoder.layer[i]
        # attention LN
        quant.intermediate.dense.weight.data *= fp32.attention.output.LayerNorm.weight.data.detach().clone()
        quant.output.delay_ln = torch.nn.Parameter(fp32.attention.output.LayerNorm.weight.data.detach().clone())
        # ffn LN
        if i > 0:
            ln = fp32_model.roberta.encoder.layer[i-1].output.LayerNorm
        else:
            ln = fp32_model.roberta.embeddings.LayerNorm
        quant.attention.self.query.weight.data *= ln.weight.data.detach().clone()
        quant.attention.self.key.weight.data *= ln.weight.data.detach().clone()
        quant.attention.self.value.weight.data *= ln.weight.data.detach().clone()
        quant.attention.output.delay_ln = torch.nn.Parameter(ln.weight.data.detach().clone())


def delay_bert_ln(fp32_model, model):
    for i in range(12):
        fp32 = fp32_model.bert.encoder.layer[i]
        quant = model.bert.encoder.layer[i]
        # attention LN
        quant.intermediate.dense.weight.data *= fp32.attention.output.LayerNorm.weight.data.detach().clone()
        quant.output.delay_ln = torch.nn.Parameter(fp32.attention.output.LayerNorm.weight.data.detach().clone())
        # ffn LN
        if i > 0:
            ln = fp32_model.bert.encoder.layer[i-1].output.LayerNorm
        else:
            ln = fp32_model.bert.embeddings.LayerNorm
        quant.attention.self.query.weight.data *= ln.weight.data.detach().clone()
        quant.attention.self.key.weight.data *= ln.weight.data.detach().clone()
        quant.attention.self.value.weight.data *= ln.weight.data.detach().clone()
        quant.attention.output.delay_ln = torch.nn.Parameter(ln.weight.data.detach().clone())


def quantize_model(fp32_model, config_quant, config_model):
    if not hasattr(config_quant, 'backend'):
        config_quant.backend = 'academic'
    if not hasattr(config_quant, 'except_quantizer'):
        config_quant.except_quantizer = None
    if not hasattr(config_quant, 'is_remove_padding'):
        config_quant.is_remove_padding = False
    if not hasattr(config_quant, 'cosine'):
        config_quant.cosine = False
    if not hasattr(config_quant, 'outlier'):
        config_quant.outlier = False
    if not hasattr(config_quant, 'delay_ln'):
        config_quant.delay_ln = False
    if not hasattr(config_quant, 'mse'):
        config_quant.mse = -1
    fp32_model.eval()
    if 'BertForSequenceClassification' in fp32_model.__class__.__name__:
        config_model.type = 'bert'
        if config_quant.delay_ln:
            model = QuantizedBertForSequenceClassificationLN(
                fp32_model, config_quant.w_qconfig,
                config_quant.a_qconfig, qoutput=False,
                backend=config_quant.backend)
            delay_bert_ln(fp32_model, model)
        else:
            model = QuantizedBertForSequenceClassification(
                fp32_model, config_quant.w_qconfig,
                config_quant.a_qconfig, qoutput=False,
                backend=config_quant.backend)
    elif 'RobertaForSequenceClassification' in fp32_model.__class__.__name__:
        config_model.type = 'roberta'
        if config_quant.delay_ln:
            model = QuantizedRobertaForSequenceClassificationLN(
                fp32_model, config_quant.w_qconfig,
                config_quant.a_qconfig, qoutput=False,
                backend=config_quant.backend)
            delay_roberta_ln(fp32_model, model)
        else:
            model = QuantizedRobertaForSequenceClassification(
                fp32_model, config_quant.w_qconfig,
                config_quant.a_qconfig, qoutput=False,
                backend=config_quant.backend)
    model.eval()
    model.is_remove_padding = config_quant.is_remove_padding
    if hasattr(config_model, 'checkpoint'):
        checkpoint = torch.load(config_model.checkpoint)
        model.load_state_dict(checkpoint)
    return model


def learn_scale(trainer, cali_data, fp_output, epoch=5):
    logger.info('*** begin learn the scale now! ***')
    para = []
    for module in trainer.model.modules():
        if isinstance(module, LSQPlusFakeQuantize):
            para.append(module.scale)
            para.append(module.zero_point)
    opt = torch.optim.Adam(para, lr=1e-5)
    iters = epoch * len(cali_data)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.)
    for i in range(epoch):
        for j, batch in enumerate(cali_data):
            opt.zero_grad()
            output = trainer.model(**batch)[0]
            loss = loss_fct(output, fp_output[j])
            loss.backward()
            opt.step()
            scheduler.step()
    torch.cuda.empty_cache()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config_path):
    config = glue_utils.parse_config(config_path)
    set_seed(config.train.seed)
    if config.data.task_name == 'cola':
        config.progress.metric_for_best_model = 'matthews_correlation'
    elif config.data.task_name == 'stsb':
        config.progress.metric_for_best_model = 'pearson'
    else:
        config.progress.metric_for_best_model = 'accuracy'
    training_args = make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets, num_labels, label_list = glue_utils.load_dataset_labels(config.data)
    tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)

    # label2id & id2label
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and config.data.task_name is not None
        and config.data.task_name != 'stsb'
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    elif config.data.task_name is not None and config.data.task_name != 'stsb':
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant, config.model)

    # max_seq_length
    config.data.max_seq_length = min(config.data.max_seq_length, tokenizer.model_max_length)

    # work with datasets, preprocess first then get train/val/test one
    raw_datasets = glue_utils.preprocess_dataset(config.data, training_args, raw_datasets, label_to_id, tokenizer)
    # train_dataset, val_dataset, predict_datasets
    train_datasets = glue_utils.check_return_data(raw_datasets, 'train',
                                                  training_args.do_train | hasattr(config, 'quant'), config.data.max_train_samples)
    if config.data.task_name == 'mnli':
        eval_datasets = (
            glue_utils.check_return_data(raw_datasets, 'validation_matched',
                                         training_args.do_eval | (hasattr(config, 'quant') and config.quant.cosine), config.data.max_eval_samples),
            glue_utils.check_return_data(raw_datasets, 'validation_mismatched',
                                         training_args.do_eval | (hasattr(config, 'quant') and config.quant.cosine), config.data.max_eval_samples),
        )
        predict_datasets = (
            glue_utils.check_return_data(raw_datasets, 'test_matched', training_args.do_predict, config.data.max_predict_samples),
            glue_utils.check_return_data(raw_datasets, 'test_mismatched', training_args.do_predict, config.data.max_predict_samples),
        )  # noqa: F841
    else:
        eval_datasets = glue_utils.check_return_data(raw_datasets, 'validation', training_args.do_eval | (hasattr(config, 'quant') and config.quant.cosine), config.data.max_eval_samples)
        predict_datasets = glue_utils.check_return_data(raw_datasets, 'test', training_args.do_predict, config.data.max_predict_samples)  # noqa: F841

    metric = load_metric("glue", config.data.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if config.data.is_regression else np.argmax(preds, axis=1)

        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if config.data.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    if training_args.do_eval:
        # used for mnli training
        trainer_eval_dataset = eval_datasets[0] if isinstance(eval_datasets, tuple) else eval_datasets
    else:
        trainer_eval_dataset = None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=trainer_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if hasattr(config, 'quant') and not hasattr(config.model, 'checkpoint'):
        # calibrate the model first
        calibrate_datasets = train_datasets.shuffle().select(range(config.quant.calibrate))
        data_loader = trainer.get_eval_dataloader(calibrate_datasets)
        enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        trainer.evaluate(calibrate_datasets.select(range(2)))
        if config.quant.a_qconfig.observer == 'EMAPruneMinMaxObserver':
            cali_data, fp32_output = get_cali_data(trainer.model, data_loader)
            for name, module in trainer.model.named_modules():
                if isinstance(module, QuantizeBase):
                    if 'act' in name:
                        module.observer.set_name(name)
                    if 'weight' in name:
                        module.disable_observer()
            evaluate_token(
                trainer.model, cali_data, fp32_output,
                config.quant.a_qconfig.bit, training_args.do_train)
            if not training_args.do_train and config.quant.a_qconfig.quantizer == 'LSQPlusFakeQuantize':
                enable_quantization(trainer.model)
                learn_scale(trainer, cali_data, fp32_output)
        else:
            enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
            trainer.evaluate(calibrate_datasets)
        torch.cuda.empty_cache()

    if training_args.do_train:
        if hasattr(config, 'quant'):
            enable_calibration_quantization(trainer.model, except_quantizer=analysis_quantize(
                config.quant.except_quantizer, config.model.type))
        train(trainer, training_args)

    if training_args.do_eval:
        if hasattr(config, 'quant'):
            enable_quantization(trainer.model, except_quantizer=analysis_quantize(
                config.quant.except_quantizer, config.model.type))
        evaluate(trainer, eval_datasets)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
