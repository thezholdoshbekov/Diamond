import torch
import torch.nn.functional as F


def compute_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    codebook_weights: torch.Tensor,
    label_smoothing: float = 0.1,
    z_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Per-codebook weighted cross-entropy with label smoothing and PAD masking.

    logits:           [B, T, K, V] — V is decoder vocab_size (logit space, no PAD/BOS)
    target_tokens:    [B, T, K] — values in [0, V) for valid positions, V (PAD) elsewhere
    target_mask:      [B, T, K] — True where target is valid (non-PAD)
    codebook_weights: [K]
    z_loss_weight:    coefficient for z-loss (PaLM, Chowdhery 2022): penalizes
                      log² of the softmax partition → keeps logits from
                      inflating, directly countering output-head/readout
                      runaway. 0 disables.
    """
    B, T, K, V = logits.shape

    flat_logits = logits.reshape(-1, V)
    flat_targets = target_tokens.reshape(-1)
    flat_mask = target_mask.reshape(-1)

    safe_targets = torch.where(flat_mask, flat_targets, torch.zeros_like(flat_targets))

    ce = F.cross_entropy(
        flat_logits,
        safe_targets,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    ce = ce * flat_mask.float()
    ce = ce.reshape(B, T, K)

    weighted = ce * codebook_weights.view(1, 1, K).to(ce.dtype)

    valid_count = target_mask.float().sum().clamp(min=1.0)
    loss = weighted.sum() / valid_count

    # z-loss (PaLM): mean of (logsumexp logits)² over valid positions. Computed
    # in fp32 for stability under bf16. Pulls the softmax normalizer toward 1,
    # preventing the logit/output-head explosion that otherwise spikes gradients.
    z_loss = logits.new_zeros(())
    if z_loss_weight > 0.0:
        lse = torch.logsumexp(flat_logits.float(), dim=-1)          # [B*T*K]
        z_per = (lse * lse) * flat_mask.float()
        z_loss = z_loss_weight * (z_per.sum() / valid_count)
        loss = loss + z_loss.to(loss.dtype)

    metrics: dict = {"z_loss": z_loss.detach() if torch.is_tensor(z_loss) else z_loss}
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == target_tokens) & target_mask
        per_cb_total = target_mask.float().sum(dim=(0, 1)).clamp(min=1.0)
        per_cb_acc = correct.float().sum(dim=(0, 1)) / per_cb_total
        metrics["per_codebook_acc"] = per_cb_acc.detach()
        metrics["acc_mean"] = per_cb_acc.mean().detach()
        metrics["loss_unweighted"] = (ce.sum() / valid_count).detach()
        # Per-codebook unweighted loss — exposes which RVQ layer regresses
        # first when total loss bounces (cb_8 first vs cb_0 first = different fixes).
        per_cb_loss = ce.sum(dim=(0, 1)) / per_cb_total
        metrics["per_codebook_loss"] = per_cb_loss.detach()

        # Per-codebook Token Error Rate = 1 - accuracy (token-level)
        metrics["per_codebook_ter"] = (1.0 - per_cb_acc).detach()

        # Logits statistics per codebook — saturation / degeneracy detection.
        # logits_max: extreme value detection (>>20 = numerical danger)
        # logits_range: discriminative power (low = uniform predictions)
        # softmax_max_prob: confidence (low = degenerate stuck state)
        # softmax_entropy: uniformity (high = model in degenerate state)
        l_f32 = logits.float()
        per_cb_max_logit = l_f32.abs().amax(dim=(0, 1, 3))      # [K]
        per_cb_logit_range = (l_f32.amax(dim=-1) - l_f32.amin(dim=-1)).mean(dim=(0, 1))  # [K]
        probs = F.softmax(l_f32, dim=-1)
        per_cb_max_prob = probs.amax(dim=-1).mean(dim=(0, 1))   # [K]
        # Softmax entropy averaged over valid positions
        entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)  # [B, T, K]
        # Mask-aware mean
        e_sum_per_cb = (entropy * target_mask.float()).sum(dim=(0, 1))
        per_cb_entropy = e_sum_per_cb / per_cb_total
        metrics["per_codebook_max_logit"] = per_cb_max_logit.detach()
        metrics["per_codebook_logit_range"] = per_cb_logit_range.detach()
        metrics["per_codebook_max_prob"] = per_cb_max_prob.detach()
        metrics["per_codebook_entropy"] = per_cb_entropy.detach()

    return loss, metrics
