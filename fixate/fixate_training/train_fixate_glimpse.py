#!/usr/bin/env python3
"""Data split, K-fold CV for hyperparameters, final train and test."""

import os
import sys
import json
from pathlib import Path

# Repo root on sys.path so `config` and `fixate` resolve from any CWD
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import product

from config.glimpse_config import *
from fixate.probing_operator.rollout_attention import (
    EXPANDED_DIRS,
    GazeDataset,
    PromptBasisModule,
    compute_option_ratios,
    gaze_distribution_from_dwell_time,
    load_gaze_data,
    load_movie_layout_from_item_features,
    load_user_features,
    slot_to_letter,
    _load_image,
    _preprocess_image_for_internvl,
)
from fixate.probing_operator.glimpse_attention import (
    GlimpseAttentionMixin,
    compute_freegeneration_attention,
    compute_baseline_freegeneration_attention,
    compute_baseline_glimpse_attention,
    compute_option_logprobs,
    define_slot_bboxes,
    normalize_gen_to_letter,
    _build_internvl_embeds,
)
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer


# ═══════════════════════════════════════════════════════════════
#  GlimpseTrainer
# ═══════════════════════════════════════════════════════════════

class GlimpseTrainer(GlimpseAttentionMixin):

    def __init__(self, model_name=MODEL_NAME, model_type=MODEL_TYPE,
                 prompt_basis=None):
        self.model_type = model_type

        if model_type == "internvl":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            self.processor = None
            self.model = AutoModel.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                device_map={"": 0} if torch.cuda.is_available() else None,
                trust_remote_code=True, use_flash_attn=False).to(DEVICE)
            self._internvl_num_image_token = self.model.num_image_token
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                device_map={"": 0} if torch.cuda.is_available() else None,
                trust_remote_code=True, attn_implementation="eager").to(DEVICE)

        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)

        if prompt_basis is None:
            hd = getattr(self.model.config, 'hidden_size', None)
            if hd is None and hasattr(self.model.config, 'text_config'):
                hd = getattr(self.model.config.text_config, 'hidden_size', None)
            if hd is None:
                try: hd = self.model.get_input_embeddings().weight.shape[-1]
                except: hd = 3584
            prompt_basis = PromptBasisModule(num_basis=NUM_BASIS, num_soft_tokens=NUM_SOFT_TOKENS, hidden_dim=hd)

        self.prompt_basis = prompt_basis.to(dtype=next(self.model.parameters()).dtype)
        self.config = TrainerConfig()
        self.slot_bboxes = define_slot_bboxes(image_width=UI_IMAGE_WIDTH, image_height=UI_IMAGE_HEIGHT)
        self.head_temp, self.depth_temp, self.eps = HEAD_TEMP, DEPTH_TEMP, 1e-8
        self._answer_token_ids = [self.tokenizer.encode(chr(ord('A')+i), add_special_tokens=False)[0]
                                  for i in range(NUM_SLOTS)]
        self.global_step = 0

    # ── Model forward ──

    def _model_forward(self, inputs, **kw):
        if self.model_type == "internvl":
            return self.model.language_model(**{k: v for k, v in inputs.items()}, **kw)
        return self.model(**inputs, **kw)

    # ── InternVL text builder ──

    def _build_internvl_text(self, profile_text, image, answer_text=None):
        instruction = INSTRUCTION
        user_prompt = f"<image>\n{profile_text}\n\n{instruction}"
        pv_cpu, np_, ar = _preprocess_image_for_internvl(image)
        sys.path.insert(0, INTERNVL_MODEL_PATH)
        from conversation import get_conv_template #conversation is shipped inside the InternVL model directory
        sys.path.pop(0)
        tmpl = get_conv_template(self.model.template)
        tmpl.system_message = "You are a sophisticated user behavior emulator."
        tmpl.append_message(tmpl.roles[0], user_prompt)
        tmpl.append_message(tmpl.roles[1], None)
        query = tmpl.get_prompt().replace(
            '<image>', '<img>' + '<IMG_CONTEXT>'*self._internvl_num_image_token*np_ + '</img>', 1)
        if answer_text is not None: query += str(answer_text)
        return query, pv_cpu, np_, ar

    # ── Build inputs with soft prompt ──

    def _build_inputs_internvl(self, image, profile_text, soft_prompt_embeds, answer_text=None):
        full_text, pv_cpu, np_, ar = self._build_internvl_text(profile_text, image, answer_text)
        tok = self.tokenizer(full_text, return_tensors="pt")
        ids, mask = tok["input_ids"], tok["attention_mask"]
        emb = self.model.get_input_embeddings(); ed, edt = emb.weight.device, emb.weight.dtype
        ids, mask = ids.to(ed), mask.to(ed)

        labels, atids, apo, apc = None, None, None, None
        if answer_text is not None:
            atids = self.tokenizer.encode(str(answer_text), add_special_tokens=False)
            m = len(atids); Lf = ids.shape[1]; ast = Lf - m
            labels = torch.full_like(ids, -100); labels[0, ast:] = ids[0, ast:]
            apo = list(range(ast, Lf))

        te = emb(ids).clone()
        pv = pv_cpu.to(device=ed, dtype=edt)
        with torch.no_grad(): vit = self.model.extract_feature(pv)
        B, N, C = te.shape; flat = te.reshape(B*N, C)
        sel = (ids.reshape(B*N) == self.model.img_context_token_id)
        vf = vit.reshape(-1, C).to(flat.device); nt = min(int(sel.sum()), vf.shape[0])
        if nt > 0: flat[sel.nonzero(as_tuple=True)[0][:nt]] = vf[:nt].detach()
        te = flat.reshape(B, N, C)
        del pv, vit, vf
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        sl = soft_prompt_embeds.shape[0]
        sp = soft_prompt_embeds.to(device=ed, dtype=te.dtype)
        ce = torch.cat([sp.unsqueeze(0), te], dim=1)
        sm = torch.ones(1, sl, dtype=torch.long, device=ed)
        inp = {"inputs_embeds": ce, "attention_mask": torch.cat([sm, mask], dim=1)}
        if labels is not None:
            inp["labels"] = torch.cat([torch.full((1, sl), -100, dtype=labels.dtype, device=ed), labels.to(ed)], 1)
            apc = [p + sl for p in apo]
        oi = {"input_ids": ids, "attention_mask": mask, "_internvl_num_patches": np_, "_internvl_aspect_ratio": ar}
        meta = {"orig_inputs": oi, "orig_input_ids": ids, "soft_len": sl,
                "answer_token_ids": atids, "answer_pos_orig": apo, "answer_pos_combined": apc, "emb_device": ed}
        return inp, meta

    def build_inputs_with_soft_prompt(self, image, profile_text, soft_prompt_embeds, answer_text=None):
        if self.model_type == "internvl":
            return self._build_inputs_internvl(image, profile_text, soft_prompt_embeds, answer_text)
        # Qwen path
        instruction = INSTRUCTION
        user_prompt = f"{profile_text}\n\n{instruction}"
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}]
        pt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = pt if answer_text is None else pt + str(answer_text)
        oi = self.processor(text=full_text, images=image, return_tensors="pt")
        ids, mask = oi["input_ids"], oi["attention_mask"]

        labels, atids, apo, apc = None, None, None, None
        if answer_text is not None:
            atids = self.tokenizer.encode(str(answer_text), add_special_tokens=False)
            m = len(atids); Lf = ids.shape[1]; ast = Lf - m
            labels = torch.full_like(ids, -100); labels[0, ast:] = ids[0, ast:]
            apo = list(range(ast, Lf))

        emb = self.model.get_input_embeddings(); ed = emb.weight.device
        ids, mask = ids.to(ed), mask.to(ed)
        if labels is not None: labels = labels.to(ed)
        for k, v in list(oi.items()):
            if torch.is_tensor(v): oi[k] = v.to(ed)
        te = emb(ids)
        sl = soft_prompt_embeds.shape[0]
        sp = soft_prompt_embeds.to(device=ed, dtype=te.dtype)
        ce = torch.cat([sp.unsqueeze(0), te], dim=1)
        sm = torch.ones(1, sl, dtype=torch.long, device=ed)
        inp = dict(oi); inp["attention_mask"] = torch.cat([sm, mask], 1)
        inp.pop("input_ids", None); inp["inputs_embeds"] = ce
        if labels is not None:
            inp["labels"] = torch.cat([torch.full((1, sl), -100, dtype=labels.dtype, device=ed), labels], 1)
            apc = [p + sl for p in apo]
        meta = {"orig_inputs": oi, "orig_input_ids": ids, "soft_len": sl,
                "answer_token_ids": atids, "answer_pos_orig": apo, "answer_pos_combined": apc, "emb_device": ed}
        return inp, meta

    # ── Loss computation ──

    def compute_losses(self, model_attn, gaze_dists, user_ids, lambda_attn_weight, beta_reg, loss_choice=None):
        g, a = gaze_dists, model_attn
        cfg, eps = self.config, self.eps
        w = ((g + cfg.power_eps).pow(cfg.power_gamma))
        w = w / w.sum(1, keepdim=True)
        la = (w * g * (torch.log(g + eps) - torch.log(a + eps))).sum(1).mean()
        lr = self.prompt_basis.get_user_alpha_l2_loss(user_ids)
        if loss_choice is None:
            lc = torch.zeros((), device=la.device, dtype=la.dtype)
        else:
            lc = loss_choice
        total = lc + lambda_attn_weight * la + beta_reg * lr
        return total, {k: float(v.detach().cpu()) for k, v in
                       {"loss_choice": lc, "loss_attn": la, "loss_reg": lr, "total_loss": total}.items()}

    # ── Shared: optimizer step ──

    def _optimizer_step(self, optimizer, user_ids):
        optimizer.step()
        self.global_step += 1

    # ── Shared: post-GLIMPSE aggregation ──

    def _finalize_losses(self, all_attn, gaze_dists, user_ids, bs, correct, loss_choice=None):
        ab = torch.stack(all_attn, dim=0)
        gaze_dists = gaze_dists.to(ab.device)
        law = self.config.lambda_attn_weight
        total, ld = self.compute_losses(ab, gaze_dists, user_ids, law, self.config.beta_reg, loss_choice=loss_choice)
        ld["lambda_attn_weight"] = float(law)
        ld["choice_acc"] = correct / bs if bs > 0 else 0.0
        return total, ld

    # ── Single-sample train step ──

    def _process_single_sample(self, i, images, profile_texts, soft_prompts, choice_slots,
                               all_attn, correct, choice_losses):
        image, pt, sp = images[i], profile_texts[i], soft_prompts[i]
        ts = max(1, min(15, int(choice_slots[i].item()) + 1))
        at = slot_to_letter(ts)
        inp, meta = self.build_inputs_with_soft_prompt(image, pt, sp, at)
        ed = meta["emb_device"]; sp = sp.to(ed)

        with torch.enable_grad():
            out = self._model_forward(inp, output_attentions=True, use_cache=False, return_dict=True)
            if self.config.train_choice and getattr(out, "loss", None) is not None:
                choice_losses.append(out.loss)
            logits, attns = out.logits, out.attentions
            ap, ai = meta["answer_pos_combined"], meta["answer_token_ids"]

            if self.config.train_choice and ap and len(ap) > 0:
                prev = ap[0] - 1
                if 0 <= prev < logits.shape[1]:
                    al15 = logits[0, prev, self._answer_token_ids]
                    if int(al15.argmax().item()) == ts-1: correct[0] += 1

            target, src = self._compute_target_scalar(logits, ap, ai)
            del logits, out; torch.cuda.empty_cache()

            need_2nd = bool(DIFF_GLIMPSE and GLIMPSE_CREATE_GRAPH)
            E, G = [], []
            L = len(attns)
            lids = list(range(L))
            needed = set(lids); al = list(attns)
            for li in range(L):
                if li not in needed and torch.is_tensor(al[li]): al[li] = None
            del attns; torch.cuda.empty_cache()

            valid = [(l, al[l]) for l in lids if al[l] is not None and torch.is_tensor(al[l]) and al[l].requires_grad]
            if valid:
                _, va = zip(*valid)
                grads = torch.autograd.grad(target, va, retain_graph=need_2nd,
                                            create_graph=(DIFF_GLIMPSE and GLIMPSE_CREATE_GRAPH), allow_unused=True)
                for (_, a), g in zip(valid, grads):
                    if g is None: continue
                    E.append(self.stage1_layer_relevance(a[0], g[0])); G.append(g[0])
            del al; torch.cuda.empty_cache()

            if not E:
                ma = sp.sum() * 0.0 + torch.ones(NUM_SLOTS, device=sp.device, dtype=sp.dtype) / NUM_SLOTS
            else:
                R = self.stage2_adaptive_propagation(E, G)
                src = max(0, min(int(src), R.shape[0]-1))
                vis = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])
                if not vis:
                    ma = torch.ones(NUM_SLOTS, device=DEVICE) / NUM_SLOTS
                else:
                    ma = self._resolve_slot_attention(R[src], vis, meta, image)

            if SLOT_TEMPERATURE < 1.0:
                ma = torch.softmax(torch.log(ma + self.eps) / SLOT_TEMPERATURE, dim=0)
            all_attn.append(ma)
        del inp, meta
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def train_step(self, batch, optimizer, accumulation_steps=1, zero_grad=True, do_optimizer_step=True):
        uids, imgs, pts = batch["user_id"], batch["image"], batch["profile_text"]
        cs, gd = batch["choice_slot"].to(DEVICE), batch["gaze_dist"].to(DEVICE)
        bs = len(uids); sps = self.prompt_basis(uids)
        ama, correct, cl = [], [0], []
        for i in range(bs):
            self._process_single_sample(i, imgs, pts, sps, cs, ama, correct, cl)
        loss_choice = torch.stack(cl).mean() if (self.config.train_choice and cl) else None
        total, ld = self._finalize_losses(ama, gd, uids, bs, correct[0], loss_choice=loss_choice)
        if zero_grad: optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        (total / accumulation_steps).backward()
        if do_optimizer_step: self._optimizer_step(optimizer, uids)
        return ld

    # ── Batched train step ──

    def train_step_batched(self, batch, optimizer, accumulation_steps=1, zero_grad=True, do_optimizer_step=True):
        uids, imgs, pts = batch["user_id"], batch["image"], batch["profile_text"]
        cs, gd = batch["choice_slot"].to(DEVICE), batch["gaze_dist"].to(DEVICE)
        bs = len(uids); sps = self.prompt_basis(uids)

        per_inp, per_meta, ats = [], [], []
        for i in range(bs):
            ts = max(1, min(15, int(cs[i].item()) + 1)); at = slot_to_letter(ts); ats.append(at)
            inp_i, meta_i = self.build_inputs_with_soft_prompt(imgs[i], pts[i], sps[i], at)
            per_inp.append(inp_i); per_meta.append(meta_i)

        # Qwen: pre-extract visual features
        if self.model_type != "internvl":
            ipad = self._find_special_id("<|image_pad|>")
            if ipad is None: return self.train_step(batch, optimizer, accumulation_steps, zero_grad, do_optimizer_step)
            for i in range(bs):
                pv = per_inp[i].pop("pixel_values", None); gthw = per_inp[i].pop("image_grid_thw", None)
                for k in ("pixel_values_videos", "video_grid_thw"): per_inp[i].pop(k, None)
                if pv is not None:
                    vd = self.model.visual.get_dtype() if hasattr(self.model, "visual") and hasattr(self.model.visual, "get_dtype") else pv.dtype
                    with torch.no_grad():
                        ie = self.model.visual(pv.to(vd), grid_thw=gthw)
                        if isinstance(ie, (tuple, list)): ie = ie[0]
                        if hasattr(ie, "last_hidden_state"): ie = ie.last_hidden_state
                        if torch.is_tensor(ie) and ie.dim() == 3 and ie.shape[0] == 1: ie = ie[0]
                    oi = per_meta[i]["orig_input_ids"][0]; sl = per_meta[i]["soft_len"]
                    emb = per_inp[i]["inputs_embeds"]; mo = (oi == ipad)
                    mc = torch.zeros(emb.shape[1], dtype=torch.bool, device=mo.device)
                    mc[sl:sl+mo.shape[0]] = mo
                    nt = min(int(mc.sum()), ie.shape[0] if torch.is_tensor(ie) else 0)
                    if nt > 0: emb[0, mc.nonzero(as_tuple=True)[0][:nt]] = ie[:nt].detach().to(emb.dtype)
                    del pv, ie
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        ed = per_meta[0]["emb_device"]; edt = per_inp[0]["inputs_embeds"].dtype
        edim = per_inp[0]["inputs_embeds"].shape[-1]
        slens = [inp["inputs_embeds"].shape[1] for inp in per_inp]; ml = max(slens)
        be = torch.zeros(bs, ml, edim, device=ed, dtype=edt)
        bm = torch.zeros(bs, ml, dtype=torch.long, device=ed)
        bl = torch.full((bs, ml), -100, dtype=torch.long, device=ed)
        for i in range(bs):
            Li = slens[i]; be[i,:Li] = per_inp[i]["inputs_embeds"][0]
            bm[i,:Li] = per_inp[i]["attention_mask"][0]
            if "labels" in per_inp[i]: bl[i,:Li] = per_inp[i]["labels"][0]

        bi = {"inputs_embeds": be, "attention_mask": bm}
        if self.config.train_choice: bi["labels"] = bl

        with torch.enable_grad():
            out = self._model_forward(bi, output_attentions=True, use_cache=False, return_dict=True)
            loss_choice = None
            if self.config.train_choice and getattr(out, "loss", None) is not None:
                loss_choice = out.loss
            logits, attns = out.logits, out.attentions
            tscalars, src_list, correct = [], [], 0
            for i in range(bs):
                m = per_meta[i]; ap, ai = m["answer_pos_combined"], m["answer_token_ids"]
                ts = max(1, min(15, int(cs[i].item())+1)); ti = ts - 1
                if self.config.train_choice and ap and len(ap) > 0:
                    prev = ap[0] - 1
                    if 0 <= prev < logits.shape[2]:
                        al15 = logits[i, prev, self._answer_token_ids]
                        if int(al15.argmax().item()) == ti: correct += 1
                sc, sp = self._compute_target_scalar(logits, ap, ai, bi=i)
                tscalars.append(sc); src_list.append(sp)

            ss = sum(tscalars)
            L = len(attns)
            lids = list(range(L))
            valid = [(l, attns[l]) for l in lids if torch.is_tensor(attns[l]) and attns[l].requires_grad]
            n2 = bool(DIFF_GLIMPSE and GLIMPSE_CREATE_GRAPH)
            all_grads = None
            if valid:
                _, va = zip(*valid)
                all_grads = torch.autograd.grad(ss, va, retain_graph=n2,
                                                create_graph=(DIFF_GLIMPSE and GLIMPSE_CREATE_GRAPH), allow_unused=True)

            ama = []
            for i in range(bs):
                m = per_meta[i]; sp = sps[i].to(ed); img = imgs[i]; src = src_list[i]
                E, G = [], []
                if all_grads:
                    for (_, a), g in zip(valid, all_grads):
                        if g is None: continue
                        E.append(self.stage1_layer_relevance(a[i], g[i])); G.append(g[i])
                if not E:
                    ma = sp.sum()*0.0 + torch.ones(NUM_SLOTS, device=sp.device, dtype=sp.dtype)/NUM_SLOTS
                else:
                    R = self.stage2_adaptive_propagation(E, G)
                    src = max(0, min(int(src), R.shape[0]-1))
                    vis = self.find_visual_token_indices(m["orig_input_ids"], m["soft_len"])
                    ma = self._resolve_slot_attention(R[src], vis, m, img) if vis else torch.ones(NUM_SLOTS, device=DEVICE)/NUM_SLOTS
                if SLOT_TEMPERATURE < 1.0: ma = torch.softmax(torch.log(ma+self.eps)/SLOT_TEMPERATURE, dim=0)
                ama.append(ma)

        total, ld = self._finalize_losses(ama, gd, uids, bs, correct, loss_choice=loss_choice)
        if zero_grad: optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        (total / accumulation_steps).backward()
        if do_optimizer_step: self._optimizer_step(optimizer, uids)
        return ld

    def save_checkpoint(self, path):
        torch.save({'prompt_basis': self.prompt_basis.state_dict(), 'global_step': self.global_step}, path)

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.prompt_basis.load_state_dict(ck['prompt_basis'], strict=False)
        self.global_step = ck.get('global_step', 0)


# ═══════════════════════════════════════════════════════════════
#  Teacher-forcing validation / CV primary score (same idea as Rollout/AttnLRP)
# ═══════════════════════════════════════════════════════════════

def evaluate_tf(trainer, samples_list, user_features, poster_dir, expanded_dirs=None,
                use_soft_prompt=True):
    all_m, per_user = [], defaultdict(list)
    n_skip, r3, r5 = 0, 0, 0
    for sample in samples_list:
        uid, tid, choice, dwell = sample["user_id"], sample["task_id"], sample["choice_slot"], sample["dwell_time"]
        image = _load_image(uid, tid, poster_dir, expanded_dirs)
        if image is None:
            n_skip += 1
            continue
        prof = user_features.get(uid, {})
        pt = f"Top_genre: {prof.get('Top_genre','')}, Preferred_genres: {prof.get('Preferred_genres','')}"
        at = slot_to_letter(choice)
        try:
            if use_soft_prompt:
                ma = trainer.compute_model_attention_single(uid, image, pt, at)
            else:
                ma = compute_baseline_glimpse_attention(trainer, image, pt, at)
            if torch.is_tensor(ma):
                ma = ma.detach().float().cpu().numpy()
        except Exception:
            n_skip += 1
            continue
        lp = None
        try:
            olp = compute_option_logprobs(trainer, image, pt, user_id=uid if use_soft_prompt else None)
            rk = np.argsort(olp)[::-1]
            gi = choice - 1
            if gi in rk[:3]:
                r3 += 1
            if gi in rk[:5]:
                r5 += 1
            lp = np.exp(olp)
            lp = lp / lp.sum()
        except Exception:
            pass
        gd = gaze_distribution_from_dwell_time(dwell)
        m = compute_sample_metrics(ma, gd, choice - 1, logit_probs=lp)
        all_m.append(m)
        per_user[uid].append(m)
    if not all_m:
        return {}
    result = {}
    for name in all_m[0]:
        result[f"micro_{name}"] = float(np.mean([m[name] for m in all_m]))
        result[f"macro_{name}"] = float(np.mean([np.mean([m[name] for m in ms]) for ms in per_user.values()]))
    result.update(n_samples=len(all_m), n_users=len(per_user),
                    **{f"recall@{k}": v / len(all_m) for k, v in [(3, r3), (5, r5)]})
    return result


# ═══════════════════════════════════════════════════════════════
#  Free generation: final test comparison (BL-FG / TR-FG)
# ═══════════════════════════════════════════════════════════════

def evaluate_freegen(trainer, samples_list, user_features, poster_dir, expanded_dirs=None,
                     use_soft_prompt=True):
    all_m, per_user = [], defaultdict(list)
    gt_let, gen_let = [], []
    n_skip, correct, r3, r5 = 0, 0, 0, 0
    for sample in samples_list:
        uid, tid, choice, dwell = sample["user_id"], sample["task_id"], sample["choice_slot"], sample["dwell_time"]
        image = _load_image(uid, tid, poster_dir, expanded_dirs)
        if image is None: n_skip += 1; continue
        prof = user_features.get(uid, {})
        pt = f"Top_genre: {prof.get('Top_genre','')}, Preferred_genres: {prof.get('Preferred_genres','')}"
        gt = slot_to_letter(choice)
        try:
            if use_soft_prompt:
                attn, gen = compute_freegeneration_attention(trainer, uid, image, pt)
            else:
                attn, gen = compute_baseline_freegeneration_attention(trainer, image, pt)
        except Exception:
            n_skip += 1
            continue

        if gen.strip().upper() == gt.strip().upper(): correct += 1
        lp = None
        try:
            olp = compute_option_logprobs(trainer, image, pt, user_id=uid if use_soft_prompt else None)
            rk = np.argsort(olp)[::-1]; gi = choice - 1
            if gi in rk[:3]: r3 += 1
            if gi in rk[:5]: r5 += 1
            lp = np.exp(olp); lp = lp / lp.sum()
        except: pass

        gd = gaze_distribution_from_dwell_time(dwell)
        m = compute_sample_metrics(attn, gd, choice-1, logit_probs=lp)
        all_m.append(m); per_user[uid].append(m)
        gt_let.append(gt.strip().upper()); gen_let.append(normalize_gen_to_letter(gen))

    if not all_m: return {}
    result = {}
    for name in all_m[0]:
        result[f"micro_{name}"] = float(np.mean([m[name] for m in all_m]))
        result[f"macro_{name}"] = float(np.mean([np.mean([m[name] for m in ms]) for ms in per_user.values()]))
    result.update(n_samples=len(all_m), n_users=len(per_user),
                  answer_accuracy=correct/len(all_m), **{f"recall@{k}": v/len(all_m) for k, v in [(3,r3),(5,r5)]})
    pfx = "tr_fg" if use_soft_prompt else "bl_fg"
    for k, v in compute_option_ratios(gen_let).items(): result[f"{pfx}_ratio_{k}"] = float(v)
    if not use_soft_prompt:
        for k, v in compute_option_ratios(gt_let).items(): result[f"gt_ratio_{k}"] = float(v)
    return result


# ═══════════════════════════════════════════════════════════════
#  Training pipeline
# ═══════════════════════════════════════════════════════════════

def _run_epoch(trainer, loader, optimizer):
    losses, accum = [], GRADIENT_ACCUMULATION_STEPS
    nb = len(loader)
    fn = trainer.train_step_batched if USE_BATCHED_FORWARD else trainer.train_step
    for bi, batch in enumerate(loader):
        ld = fn(batch, optimizer, accumulation_steps=accum,
                zero_grad=(bi % accum == 0), do_optimizer_step=((bi+1) % accum == 0 or bi+1 == nb))
        losses.append(ld)
    return {k: float(np.mean([d[k] for d in losses if k in d])) for k in losses[0]}


def train_one_fold(trainer, train_samps, val_samps, user_features, poster_dir, expanded_dirs,
                   hp, num_epochs, eval_every):
    apply_hyperparams(trainer, hp); reset_prompt_basis(trainer, hp)
    for uid in set(s["user_id"] for s in train_samps + val_samps):
        trainer.prompt_basis.get_or_create_user_alpha(uid)
    opt = build_optimizer(trainer, hp)
    loader = DataLoader(GazeDataset(train_samps, user_features, poster_dir, expanded_dirs=expanded_dirs),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    best_score = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_metrics, best_epoch, best_state = None, 0, None
    for epoch in range(1, num_epochs+1):
        avg = _run_epoch(trainer, loader, opt)
        do_eval = (epoch % eval_every == 0) or epoch == num_epochs
        if do_eval:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            vm = evaluate_tf(trainer, val_samps, user_features, poster_dir, expanded_dirs)
            sc, _det = compute_primary_score(vm)
            improved = np.isfinite(sc) and ((sc < best_score) if PRIMARY_LOWER_IS_BETTER else (sc > best_score))
            if improved:
                best_score, best_metrics, best_epoch = sc, vm, epoch
                best_state = {k: v.cpu().clone() for k, v in trainer.prompt_basis.state_dict().items()}
    return {"best_epoch": best_epoch, "best_val_score": best_score, "best_val_metrics": best_metrics or {}, "best_state": best_state}


def cross_validate(trainer, samples, folds, user_features, poster_dir, expanded_dirs, hp, num_epochs, eval_every):
    K = len(folds); results = []
    for f in range(K):
        ti, vi = folds[f]
        r = train_one_fold(trainer, [samples[i] for i in ti], [samples[i] for i in vi],
                           user_features, poster_dir, expanded_dirs, hp, num_epochs, eval_every)
        results.append(r)
    valid = [r for r in results if r["best_val_metrics"]]
    if not valid: return {"avg_metrics": {}, "avg_best_epoch": 0, "fold_results": results}
    avg_m = {k: float(np.mean([r["best_val_metrics"][k] for r in valid])) for k in valid[0]["best_val_metrics"]}
    avg_ep = float(np.mean([r["best_epoch"] for r in valid]))
    sc, _det = compute_primary_score(avg_m)
    return {"avg_metrics": avg_m, "avg_primary_score": sc, "avg_best_epoch": avg_ep, "fold_results": results}


def grid_search_cv(trainer, samples, folds, user_features, poster_dir, expanded_dirs, param_grid, num_epochs, eval_every):
    keys = sorted(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))
    best_score = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_hp, best_cv, log_ = None, None, []
    for combo in combos:
        hp = dict(DEFAULT_HYPERPARAMS); hp.update(zip(keys, combo))
        cv = cross_validate(trainer, samples, folds, user_features, poster_dir, expanded_dirs, hp, num_epochs, eval_every)
        sc = cv.get("avg_primary_score", float("nan"))
        log_.append({"hyperparams": {k: hp[k] for k in keys}, "primary_score": sc, "avg_metrics": cv["avg_metrics"]})
        if np.isfinite(sc) and ((sc < best_score) if PRIMARY_LOWER_IS_BETTER else (sc > best_score)):
            best_score, best_hp, best_cv = sc, hp.copy(), cv
    with open(os.path.join(OUTPUT_DIR, f"glimpse_grid_search_{RUN_TIMESTAMP}.json"), "w") as f:
        json.dump(log_, f, indent=2, ensure_ascii=False)
    return best_hp, best_cv


# ═══════════════════════════════════════════════════════════════
#  Final train & test
# ═══════════════════════════════════════════════════════════════

def final_train_and_test(trainer, all_samples, test_idx, user_features, poster_dir, expanded_dirs,
                         hp, num_epochs, val_ratio=0.0):
    test_set = set(test_idx)
    train_all = [s for i, s in enumerate(all_samples) if i not in test_set]
    test_samps = [all_samples[i] for i in test_idx]

    use_val = val_ratio > 0 and len(train_all) >= 10
    if use_val:
        rng = np.random.RandomState(SEED)
        nv = max(1, int(len(train_all) * val_ratio)); perm = rng.permutation(len(train_all))
        vi = set(perm[:nv].tolist())
        train_samps = [s for i, s in enumerate(train_all) if i not in vi]
        val_samps = [train_all[i] for i in sorted(vi)]
    else:
        train_samps, val_samps = train_all, []

    apply_hyperparams(trainer, hp); reset_prompt_basis(trainer, hp)
    for uid in set(s["user_id"] for s in all_samples):
        trainer.prompt_basis.get_or_create_user_alpha(uid)
    opt = build_optimizer(trainer, hp)
    loader = DataLoader(GazeDataset(train_samps, user_features, poster_dir, expanded_dirs=expanded_dirs),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

    best_score = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_epoch, best_state, no_imp = 0, None, 0
    for epoch in range(1, num_epochs+1):
        avg = _run_epoch(trainer, loader, opt)
        do_eval = use_val and ((epoch % EVAL_EVERY_N_EPOCHS == 0) or epoch == num_epochs)
        if do_eval:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            vm = evaluate_tf(trainer, val_samps, user_features, poster_dir, expanded_dirs)
            sc, _ = compute_primary_score(vm)
            improved = np.isfinite(sc) and ((sc < best_score) if PRIMARY_LOWER_IS_BETTER else (sc > best_score))
            if improved:
                best_score, best_epoch = sc, epoch
                best_state = {k: v.cpu().clone() for k, v in trainer.prompt_basis.state_dict().items()}
                no_imp = 0
            else: no_imp += 1
            if no_imp >= 6: break

    if use_val and best_state: trainer.prompt_basis.load_state_dict(best_state)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    trainer.save_checkpoint(os.path.join(CHECKPOINT_DIR, f"glimpse_final_{RUN_TIMESTAMP}.pt"))

    bl = evaluate_freegen(trainer, test_samps, user_features, poster_dir, expanded_dirs, use_soft_prompt=False)
    tr = evaluate_freegen(trainer, test_samps, user_features, poster_dir, expanded_dirs, use_soft_prompt=True)

    result_path = os.path.join(OUTPUT_DIR, f"att_glimpse_final_{RUN_TIMESTAMP}.json")
    with open(result_path, "w") as f:
        json.dump({"baseline_fg": bl, "trained_fg": tr, "hyperparams": hp}, f, indent=2, ensure_ascii=False)
    combined = dict(tr) if tr else dict(bl)
    combined["_fg_pair_metrics"] = {"bl_fg": bl, "tr_fg": tr}
    return combined


def multi_seed_final(trainer, all_samples, test_idx, user_features, poster_dir, expanded_dirs,
                     hp, num_epochs, val_ratio=0.0):
    seeds = MULTI_SEEDS
    all_m, all_fg = [], []
    best_score = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf")
    best_seed = None
    for seed in seeds:
        set_all_seeds(seed)
        m = final_train_and_test(trainer, all_samples, test_idx, user_features, poster_dir, expanded_dirs,
                                 hp, num_epochs, val_ratio=val_ratio)
        all_m.append((seed, m))
        fg = m.pop("_fg_pair_metrics", None)
        if fg: all_fg.append((seed, fg))
        sc, _ = compute_primary_score(m)
        if np.isfinite(sc) and ((sc < best_score) if PRIMARY_LOWER_IS_BETTER else (sc > best_score)):
            best_score, best_seed = sc, seed

    avg = {}
    all_keys = set(); [all_keys.update(m.keys()) for _, m in all_m]
    for k in sorted(all_keys):
        vals = [m[k] for _, m in all_m if k in m and isinstance(m[k], (int, float)) and np.isfinite(m[k])]
        if vals: avg[k] = float(np.mean(vals))
    return avg, all_fg


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    uf = load_user_features(USER_FEATURES_CSV)
    ml = load_movie_layout_from_item_features(ITEM_FEATURES_CSV, task_id_min=1, task_id_max=35, max_movie_pos=15)
    samples = load_gaze_data(GAZE_DATA_CSV, ml)
    expanded_dirs = EXPANDED_DIRS
    samples = [s for s in samples if _load_image(s["user_id"], s["task_id"], POSTER_IMAGES_DIR, expanded_dirs)]
    test_idx, folds = split_leave1_test_then_kfold(samples, K=K_FOLDS, seed=SEED)
    trainer = GlimpseTrainer(MODEL_NAME, MODEL_TYPE)
    if RUN_GRID_SEARCH:
        best_hp, best_cv = grid_search_cv(trainer, samples, folds, uf, POSTER_IMAGES_DIR, expanded_dirs,
                                           PARAM_GRID, NUM_EPOCHS_CV, EVAL_EVERY_N_EPOCHS)
        final_ep = max(1, round(best_cv.get("avg_best_epoch", NUM_EPOCHS_FINAL)))
    else:
        best_hp = dict(DEFAULT_HYPERPARAMS)
        final_ep = NUM_EPOCHS_FINAL
    fvr = NO_CV_VAL_RATIO if not RUN_GRID_SEARCH else 0.0
    if USE_MULTI_SEED:
        test_m, all_fg = multi_seed_final(trainer, samples, test_idx, uf, POSTER_IMAGES_DIR, expanded_dirs,
                                           best_hp, final_ep, val_ratio=fvr)
    else:
        set_all_seeds(SEED)
        test_m = final_train_and_test(trainer, samples, test_idx, uf, POSTER_IMAGES_DIR, expanded_dirs,
                                      best_hp, final_ep, val_ratio=fvr)


if __name__ == "__main__":
    main()