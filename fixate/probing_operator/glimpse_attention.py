"""GLIMPSE attention extraction: Mixin class + baseline/freegen utilities."""
import sys, math
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from PIL import Image
from config.glimpse_config import *


class GlimpseAttentionMixin:
    """Mixin providing all GLIMPSE attention computation methods.
    Requires host class to set: eps, head_temp, depth_temp, config, model_type,
    tokenizer, model, _internvl_num_image_token, _answer_token_ids, prompt_basis.
    """

    def _find_special_id(self, tok_str):
        try:
            tid = self.tokenizer.convert_tokens_to_ids(tok_str)
            return None if tid is None or tid == self.tokenizer.unk_token_id else tid
        except: return None

    def find_visual_token_indices(self, input_ids, soft_len):
        ids = input_ids[0].tolist()
        vis = []
        vs, ve = self._find_special_id("<|vision_start|>"), self._find_special_id("<|vision_end|>")
        if vs and ve and vs in ids and ve in ids:
            s, e = ids.index(vs), ids.index(ve)
            if e > s + 1: vis = list(range(s+1, e))
        for tok in ["<|image_pad|>", "<IMG_CONTEXT>", "<image>"]:
            if vis: break
            tid = self._find_special_id(tok)
            if tid: vis = [i for i, t in enumerate(ids) if t == tid]
        return [i + soft_len for i in vis]

    # ── GLIMPSE Stage 1 & 2 ──

    def stage1_layer_relevance(self, attn_l, grad_l):
        G = torch.relu(grad_l * attn_l)
        score = G.sum(dim=(1, 2)) / (torch.relu(grad_l).sum(dim=(1, 2)) + self.eps)
        w = torch.softmax(score / self.head_temp, dim=0)
        E = sum(w[h] * G[h] for h in range(G.shape[0]))
        return E / (E.sum(dim=-1, keepdim=True) + self.eps)

    def stage2_adaptive_propagation(self, E_list, grad_list):
        L, N = len(E_list), E_list[0].shape[0]
        g_layers = torch.stack([torch.sum(torch.abs(g.sum(dim=0))) for g in grad_list])
        s = torch.softmax(self.depth_temp * torch.arange(1, L+1, device=g_layers.device, dtype=g_layers.dtype), dim=0)
        alpha = g_layers * s; alpha = alpha / (alpha.sum() + self.eps)
        R = torch.eye(N, device=E_list[0].device, dtype=E_list[0].dtype)
        for l in range(L): R = R + (alpha[l] * E_list[l]) @ R
        return R

    # ── Patch → Slot aggregation ──

    def aggregate_patch_saliency_to_slots_torch(self, patch_sal, image_size, patch_grid_size):
        dev, dt = patch_sal.device, patch_sal.dtype
        Hp, Wp = patch_grid_size
        if patch_sal.dim() == 1:
            n, hw = patch_sal.numel(), Hp * Wp
            if n == hw: ps = patch_sal.view(Hp, Wp)
            elif hw == 4*n:
                ps = patch_sal.view(Hp//2, Wp//2).repeat_interleave(2,0).repeat_interleave(2,1)[:Hp,:Wp]
            elif n > hw: ps = patch_sal[:hw].view(Hp, Wp)
            else: pad = torch.zeros(hw, device=dev, dtype=dt); pad[:n] = patch_sal; ps = pad.view(Hp, Wp)
        else: ps = patch_sal; Hp, Wp = ps.shape
        ps = torch.relu(ps)
        iw, ih = float(image_size[0]), float(image_size[1])
        gi, gj = torch.meshgrid(torch.arange(Hp, device=dev), torch.arange(Wp, device=dev), indexing="ij")
        cx, cy = (gj.float()+0.5)*iw/Wp, (gi.float()+0.5)*ih/Hp
        xs, xe, ys, ye = iw*0.05, iw*0.95, ih*0.1, ih*0.95
        mask = (cx >= xs) & (cx < xe) & (cy >= ys) & (cy < ye)
        col = torch.clamp(((cx-xs)/((xe-xs)/NUM_COLS)).long(), 0, NUM_COLS-1)
        row = torch.clamp(((cy-ys)/((ye-ys)/NUM_ROWS)).long(), 0, NUM_ROWS-1)
        idx = (row*NUM_COLS + col).reshape(-1)
        vals = ps.reshape(-1)[mask.reshape(-1)]
        out = torch.zeros(NUM_SLOTS, device=dev, dtype=dt)
        if vals.numel() > 0: out = out.scatter_add(0, idx[mask.reshape(-1)], vals)
        return out / (out.sum() + self.eps)

    # ── Patch grid helpers ──

    def _get_internvl_patch_grid(self, ar, n_vis):
        t = int(math.sqrt(self._internvl_num_image_token))
        return (t*ar[0], t*ar[1])

    def _get_internvl_content_count(self, ar):
        t = int(math.sqrt(self._internvl_num_image_token))
        return t*ar[0] * t*ar[1]

    def _fix_patch_grid(self, pg, n_vis):
        H, W = pg; hw = H*W
        if hw == n_vis: return pg
        if hw == 4*n_vis and H%2==0 and W%2==0: return (H//2, W//2)
        if n_vis == 4*hw: return (H*2, W*2)
        side = int(math.sqrt(n_vis))
        for h in range(side, 0, -1):
            if n_vis % h == 0: return (h, n_vis//h)
        return (1, n_vis)

    def _infer_patch_grid(self, orig_inputs, n_vis):
        if self.model_type == "internvl":
            ar = orig_inputs.get("_internvl_aspect_ratio")
            if ar: return self._get_internvl_patch_grid(ar, n_vis)
        for key in ("image_grid_thw", "grid_thw", "vision_grid_thw"):
            v = orig_inputs.get(key)
            if v is not None and torch.is_tensor(v):
                thw = v.reshape(-1)
                if thw.numel() >= 3: return (int(thw[-2].item()), int(thw[-1].item()))
        side = int(math.sqrt(n_vis))
        return (max(1, side), max(1, n_vis//max(1, side)))

    # ── Shared: relevance → slot attention ──

    def _resolve_slot_attention(self, relevance, vis_indices, meta, image):
        if not vis_indices:
            return torch.ones(NUM_SLOTS, device=relevance.device, dtype=relevance.dtype) / NUM_SLOTS
        oi = meta["orig_inputs"]
        if self.model_type == "internvl":
            ar = oi.get("_internvl_aspect_ratio")
            if ar:
                cn = self._get_internvl_content_count(ar)
                if len(vis_indices) > cn: vis_indices = vis_indices[:cn]
        vr = torch.relu(relevance[vis_indices])
        pg = self._fix_patch_grid(self._infer_patch_grid(oi, len(vis_indices)), len(vis_indices))
        return self.aggregate_patch_saliency_to_slots_torch(vr, (image.width, image.height), pg)

    # ── Shared: target scalar from answer tokens ──

    def _compute_target_scalar(self, logits, ans_pos, ans_ids, bi=0):
        if not ans_pos or not ans_ids:
            return logits[bi, -1].max(), logits.shape[1]-1
        scalar, sp = None, logits.shape[1]-1
        for p, tid in zip(ans_pos, ans_ids):
            prev = p - 1
            if prev >= 0:
                v = logits[bi, prev, tid]
                scalar = v if scalar is None else scalar + v; sp = prev
        return (scalar, sp) if scalar is not None else (logits[bi, -1].max(), logits.shape[1]-1)

    # ── Shared: GLIMPSE backward ──

    def _glimpse_backward(self, target_scalar, attentions, retain=False, create=False, fp32=False, bi=0):
        L = len(attentions)
        ids = list(range(L))
        valid = [(l, attentions[l]) for l in ids if torch.is_tensor(attentions[l]) and attentions[l].requires_grad]
        if not valid: return [], []
        _, attns = zip(*valid)
        grads = torch.autograd.grad(target_scalar, attns, retain_graph=retain, create_graph=create, allow_unused=True)
        E, G = [], []
        for a, g in zip(attns, grads):
            if g is None: continue
            al, gl = a[bi], g[bi]
            if fp32: al, gl = al.float(), gl.float()
            E.append(self.stage1_layer_relevance(al, gl)); G.append(gl)
        return E, G

    # ── Single-sample eval attention (replaces patch_trainer_for_eval) ──

    def compute_model_attention_single(self, user_id, image, profile_text, answer_text=None):
        self.model.eval()
        sp = self.prompt_basis([user_id])[0]
        inputs, meta = self.build_inputs_with_soft_prompt(image=image, profile_text=profile_text,
                                                          soft_prompt_embeds=sp, answer_text=answer_text)
        with torch.enable_grad():
            out = self._model_forward(inputs, output_attentions=True, use_cache=False, return_dict=True)
            ts, sp_ = self._compute_target_scalar(out.logits, meta["answer_pos_combined"], meta["answer_token_ids"])
            E, G = self._glimpse_backward(ts, out.attentions, fp32=True)
        if not E: return torch.ones(NUM_SLOTS, dtype=torch.float32) / NUM_SLOTS
        R = self.stage2_adaptive_propagation(E, G)
        sp_ = max(0, min(sp_, R.shape[0]-1))
        vis = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])
        attn = self._resolve_slot_attention(R[sp_], vis, meta, image).detach().float()
        s = attn.sum()
        return attn / s if s > 1e-9 else torch.ones(NUM_SLOTS, dtype=torch.float32) / NUM_SLOTS


# ═══════════════════════════════════════════════════════════════
#  Standalone: baseline / free-generation attention
# ═══════════════════════════════════════════════════════════════

class _AnswerConstraint:
    def __init__(self, ids): self.ids = ids
    def __call__(self, input_ids, scores):
        m = torch.full_like(scores, float('-inf')); m[:, self.ids] = 0; return scores + m


def _generate_answer(trainer, inputs, is_internvl=False):
    safe = {k: v for k, v in inputs.items() if not k.startswith("_")}
    ilen = safe.get("inputs_embeds", safe.get("input_ids", torch.empty(1, 0))).shape[1]
    kw = dict(**safe, max_new_tokens=1, do_sample=False, pad_token_id=trainer.tokenizer.pad_token_id,
              logits_processor=[_AnswerConstraint(trainer._answer_token_ids)])
    out = (trainer.model.language_model if is_internvl else trainer.model).generate(**kw)
    tok = out[0][ilen:] if out.shape[1] > ilen else out[0]
    return trainer.tokenizer.decode(tok, skip_special_tokens=True).strip()


def _build_internvl_embeds(trainer, image, profile_text, answer_text=None, require_grad=False):
    """Build InternVL text_embeds with ViT features injected."""
    dev = next(trainer.model.parameters()).device
    emb = trainer.model.get_input_embeddings()
    txt, pv_cpu, np_, ar = trainer._build_internvl_text(profile_text, image, answer_text)
    tok = trainer.tokenizer(txt, return_tensors="pt")
    ids, mask = tok["input_ids"].to(dev), tok["attention_mask"].to(dev)
    te = emb(ids).clone()
    pv = pv_cpu.to(device=dev, dtype=te.dtype)
    with torch.no_grad(): vit = trainer.model.extract_feature(pv)
    B, N, C = te.shape; flat = te.reshape(B*N, C)
    sel = (ids.reshape(B*N) == trainer.model.img_context_token_id)
    vf = vit.reshape(-1, C).to(flat.device); nt = min(int(sel.sum()), vf.shape[0])
    if nt > 0: flat[sel.nonzero(as_tuple=True)[0][:nt]] = vf[:nt].detach()
    te = flat.reshape(B, N, C)
    del pv, vit, vf
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if require_grad: te = te.detach().requires_grad_(True)
    return te, ids, mask, ar


def _build_baseline_inputs(trainer, image, profile_text, answer_text=None, for_generate=False):
    dev = next(trainer.model.parameters()).device
    if trainer.model_type == "internvl":
        ans = None if for_generate else answer_text
        te, ids, mask, ar = _build_internvl_embeds(trainer, image, profile_text, ans, require_grad=not for_generate)
        return {"inputs_embeds": te, "attention_mask": mask, "_internvl_aspect_ratio": ar}, ids, {}
    pfx = (f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
           f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{profile_text}<|im_end|>\n"
           f"<|im_start|>assistant\n")
    text = pfx if for_generate else pfx + str(answer_text)
    inp = trainer.processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inp = {k: v.to(dev) for k, v in inp.items()}
    ids = inp["input_ids"]
    if not for_generate:
        emb = trainer.model.get_input_embeddings()
        te = emb(ids).detach().requires_grad_(True)
        inp.pop("input_ids", None); inp["inputs_embeds"] = te
    return inp, ids, {}


def _glimpse_from_inputs(trainer, inputs, input_ids, answer_text, image):
    """Core GLIMPSE extraction without soft prompt."""
    uniform = np.ones(NUM_SLOTS, dtype=np.float32) / NUM_SLOTS
    with torch.enable_grad():
        out = trainer._model_forward(inputs, output_attentions=True, use_cache=False, return_dict=True)
        Lf = input_ids.shape[1]
        slen = inputs["inputs_embeds"].shape[1] - Lf if "inputs_embeds" in inputs else 0
        ans_tids = trainer.tokenizer.encode(answer_text, add_special_tokens=False)
        ids_list = input_ids[0].tolist()
        scalar, sp = None, out.logits.shape[1]-1
        for tid in ans_tids:
            for pos in range(len(ids_list)-1, -1, -1):
                if ids_list[pos] == tid:
                    prev = pos + slen - 1
                    if prev >= 0:
                        v = out.logits[0, prev, tid]
                        scalar = v if scalar is None else scalar + v; sp = prev
                    break
        if scalar is None: scalar = out.logits[0, -1].max()
        E, G = trainer._glimpse_backward(scalar, out.attentions, fp32=True)
    if not E: return uniform
    R = trainer.stage2_adaptive_propagation(E, G)
    sp = max(0, min(sp, R.shape[0]-1)); rel = R[sp]
    ids_ = input_ids[0].tolist(); vis = []
    if trainer.model_type == "internvl":
        cid = trainer.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if cid: vis = [i for i, t in enumerate(ids_) if t == cid]
    if not vis:
        vs = trainer.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve = trainer.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        try: s, e = ids_.index(vs), ids_.index(ve, ids_.index(vs)); vis = list(range(s+1, e))
        except: pass
    if not vis: return uniform
    vis_s = [i + slen for i in vis]
    if trainer.model_type == "internvl":
        ar = inputs.get("_internvl_aspect_ratio")
        if ar:
            cn = trainer._get_internvl_content_count(ar)
            if len(vis_s) > cn: vis_s = vis_s[:cn]
    vr = torch.relu(rel[vis_s])
    pg = trainer._infer_patch_grid(inputs, len(vis_s))
    pg = trainer._fix_patch_grid(pg, len(vis_s))
    sa = trainer.aggregate_patch_saliency_to_slots_torch(vr, (image.width, image.height), pg)
    sa = sa.detach().float().cpu().numpy(); s = sa.sum()
    return sa / s if s > 1e-9 else uniform


def compute_baseline_glimpse_attention(trainer, image, profile_text, answer_text):
    trainer.model.eval()
    mi, ids, _ = _build_baseline_inputs(trainer, image, profile_text, answer_text)
    return _glimpse_from_inputs(trainer, mi, ids, answer_text, image)


def compute_freegeneration_attention(trainer, user_id, image, profile_text):
    trainer.model.eval()
    dev = next(trainer.model.parameters()).device
    emb = trainer.model.get_input_embeddings(); ed = emb.weight.device; ivl = trainer.model_type == "internvl"
    if ivl:
        te, ids, mask, _ = _build_internvl_embeds(trainer, image, profile_text)
    else:
        text = (f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{profile_text}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        inp = trainer.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        ids, mask, te = inp["input_ids"], inp["attention_mask"], emb(inp["input_ids"].to(ed))
    sp = trainer.prompt_basis([user_id])[0].to(device=ed, dtype=te.dtype)
    ge = torch.cat([sp.unsqueeze(0), te], dim=1); sl = sp.shape[0]
    gi = {"inputs_embeds": ge, "attention_mask": torch.cat([torch.ones(1, sl, dtype=torch.long, device=ed), mask.to(ed)], 1)}
    if not ivl:
        for k, v in inp.items():
            if k not in ("input_ids", "attention_mask", "inputs_embeds"): gi[k] = v
    with torch.no_grad(): gt = _generate_answer(trainer, gi, ivl)
    attn = trainer.compute_model_attention_single(user_id, image, profile_text, gt)
    if torch.is_tensor(attn): attn = attn.detach().float().cpu().numpy()
    s = float(attn.sum())
    return (attn/s if s > 1e-9 else np.ones(NUM_SLOTS, dtype=np.float32)/NUM_SLOTS), gt


def compute_baseline_freegeneration_attention(trainer, image, profile_text):
    trainer.model.eval()
    mi, ids, _ = _build_baseline_inputs(trainer, image, profile_text, for_generate=True)
    with torch.no_grad(): gt = _generate_answer(trainer, mi, trainer.model_type == "internvl")
    return compute_baseline_glimpse_attention(trainer, image, profile_text, gt), gt


def compute_option_logprobs(trainer, image, profile_text, user_id=None):
    trainer.model.eval(); ivl = trainer.model_type == "internvl"
    if user_id is not None:
        sp = trainer.prompt_basis([user_id])[0]
        inp, _ = trainer.build_inputs_with_soft_prompt(image=image, profile_text=profile_text,
                                                       soft_prompt_embeds=sp, answer_text=None)
        fwd = {k: v for k, v in inp.items() if k != "labels" and not k.startswith("_")}
    else:
        fwd, _, _ = _build_baseline_inputs(trainer, image, profile_text, for_generate=True)
        fwd = {k: v for k, v in fwd.items() if not k.startswith("_")}
    with torch.no_grad(): out = (trainer.model.language_model if ivl else trainer.model)(**fwd)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return np.array([lp[tid].item() for tid in trainer._answer_token_ids])


def define_slot_bboxes(image_width=1100, image_height=600):
    """3x5 slot rectangles (row-major) in image coordinates; matches coarse UI layout."""
    xs, xe = int(image_width * 0.05), int(image_width * 0.95)
    ys, ye = int(image_height * 0.1), int(image_height * 0.95)
    cw = max(1, (xe - xs) // NUM_COLS)
    rh = max(1, (ye - ys) // NUM_ROWS)
    boxes = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            x0, y0 = xs + c * cw, ys + r * rh
            boxes.append((x0, y0, x0 + cw, y0 + rh))
    return boxes


def normalize_gen_to_letter(s):
    if not s:
        return "OTHER"
    u = str(s).strip().upper()
    return u if len(u) == 1 and "A" <= u <= "O" else "OTHER"