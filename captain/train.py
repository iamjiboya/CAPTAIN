import torch
import torch.nn as nn
import torch.distributed as dist
import time
import warnings
import logging

logger = logging.getLogger(__name__)

def train(model: nn.Module, loader: DataLoader, config, device, optimizer, scheduler, scaler, criterion, criterion_quantile, vocab, pad_token, mask_value):
    """Train the model for one epoch."""
    model.train()
    dist.barrier()
    total_loss, total_mse, total_cls, total_cce = 0.0, 0.0, 0.0, 0.0
    total_mvc, total_ecs, total_dab, total_adv_E, total_adv_D = 0.0, 0.0, 0.0, 0.0, 0.0
    total_zero_log_prob, total_mvc_zero_log_prob = 0.0, 0.0
    total_error = 0.0
    start_time = time.time()
    log_interval = 100

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        species_values = batch_data["species_values"].to(device)
        adt_values = batch_data["adt_values"].to(device)
        adt_data = torch.arange(0, adt_values.shape[1], device=adt_values.device).repeat(adt_values.shape[0], 1)

        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict, transformer_out = model.module.rna_model(
                input_gene_ids,
                input_values,
                species_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=config.MVC,
                ECS=config.ecs_thres > 0,
                do_sample=do_sample_in_train,
            )
            adt_embeddings, adt_to_out, adt_to_out_quantiles, adt_gene_atten, labels_adt_data, adt_mask = model.module.adt_model(
                adt_data,
                transformer_out,
                src_key_padding_mask,
                adt_values,
                output_atten=False
            )
            masked_positions = input_values.eq(mask_value)
            loss_weights = {"mlm": 0.2, "adt_mse": 0.6, "adt_quantile": 0.2}
            loss_mlm = loss_weights["mlm"] * criterion(output_dict["mlm_output"], target_values, masked_positions)
            loss_adt_mse = loss_weights["adt_mse"] * criterion(adt_to_out.squeeze(-1), labels_adt_data, adt_mask)
            loss_adt_quantile = loss_weights["adt_quantile"] * criterion_quantile(adt_to_out_quantiles, labels_adt_data, adt_mask)
            loss = loss_mlm + loss_adt_mse + loss_adt_quantile

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False if scaler.is_enabled() else True)
            if len(w) > 0:
                logger.warning(f"Found infinite gradient. Scale: {scaler.get_scale()}.")
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mlm.item()
        total_cls += loss_adt_mse.item()
        total_cce += loss_adt_quantile.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval
            cur_cce = total_cce / log_interval
            print(f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | cls {cur_cls:5.2f} | cce {cur_cce:5.2f}")
            total_loss, total_mse, total_cls, total_cce = 0.0, 0.0, 0.0, 0.0
            start_time = time.time()