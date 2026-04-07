#!/usr/bin/env python3
"""AttnLRP: data split, K-fold CV, grid search, final train/test."""
import os, sys, json, math
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from itertools import product
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModel, AutoTokenizer

from config.attnlrp_config import *
from config.attnlrp_config import (
    set_all_seeds, compute_primary_score, compute_sample_metrics,
    apply_hyperparams, reset_prompt_basis, build_optimizer, collate_fn,
)
from fixate.probing_operator.attnlrp_attention import (
    EXPANDED_DIRS,
    AttnLRPAttentionMixin, PromptBasisModule, GazeDataset,
    _preprocess_image_for_internvl, slot_to_letter, _normalize_option_letter,
    compute_option_ratios, gaze_distribution_from_dwell_time,
    load_gaze_data, load_movie_layout_from_item_features, load_user_features,
    _load_image, compute_freegeneration_attention, compute_baseline_freegeneration_attention,
    compute_option_logprobs, compute_baseline_attnlrp_attention,
)


class AttnLRPTrainer(AttnLRPAttentionMixin):

    def __init__(self, model_name=MODEL_NAME, model_type=MODEL_TYPE, prompt_basis=None):
        self.model_type = model_type
        if model_type == "internvl":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            self.processor = None
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                device_map={"": 0} if torch.cuda.is_available() else None,
                trust_remote_code=True, use_flash_attn=False).to(DEVICE)
            self._internvl_num_image_token = self.model.num_image_token
            self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16,
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
        self.eps = 1e-4; self.global_step = 0
        self._answer_token_ids = [self.tokenizer.encode(chr(ord('A')+i), add_special_tokens=False)[0] for i in range(NUM_SLOTS)]

    def _model_forward(self, inputs, **kw):
        if self.model_type == "internvl": return self.model.language_model(**{k:v for k,v in inputs.items()}, **kw)
        return self.model(**inputs, **kw)

    def _build_internvl_text(self, profile_text, image, answer_text=None):
        instruction = INSTRUCTION
        pv_cpu, np_, ar = _preprocess_image_for_internvl(image)
        sys.path.insert(0, INTERNVL_MODEL_PATH); 
        from conversation import get_conv_template; sys.path.pop(0) #conversation is shipped inside the InternVL model directory
        tmpl = get_conv_template(self.model.template)
        tmpl.system_message = "You are a sophisticated user behavior emulator."
        tmpl.append_message(tmpl.roles[0], f"<image>\n{profile_text}\n\n{instruction}")
        tmpl.append_message(tmpl.roles[1], None)
        q = tmpl.get_prompt().replace('<image>', '<img>'+'<IMG_CONTEXT>'*self._internvl_num_image_token*np_+'</img>', 1)
        if answer_text is not None: q += str(answer_text)
        return q, pv_cpu, np_, ar

    def _build_inputs_internvl(self, image, profile_text, soft_prompt_embeds, answer_text=None):
        txt, pv_cpu, np_, ar = self._build_internvl_text(profile_text, image, answer_text)
        tok = self.tokenizer(txt, return_tensors="pt"); ids, mask = tok["input_ids"], tok["attention_mask"]
        emb = self.model.get_input_embeddings(); ed, edt = emb.weight.device, emb.weight.dtype
        ids, mask = ids.to(ed), mask.to(ed)
        labels, atids, apo, apc = None, None, None, None
        if answer_text is not None:
            atids = self.tokenizer.encode(str(answer_text), add_special_tokens=False)
            m = len(atids); Lf = ids.shape[1]; ast = Lf-m
            labels = torch.full_like(ids, -100); labels[0, ast:] = ids[0, ast:]; apo = list(range(ast, Lf))
        te = emb(ids).clone(); pv = pv_cpu.to(device=ed, dtype=edt)
        with torch.no_grad(): vit = self.model.extract_feature(pv)
        B,N,C = te.shape; flat = te.reshape(B*N,C)
        sel = (ids.reshape(B*N)==self.model.img_context_token_id)
        vf = vit.reshape(-1,C).to(flat.device); nt = min(int(sel.sum()), vf.shape[0])
        if nt>0: flat[sel.nonzero(as_tuple=True)[0][:nt]] = vf[:nt].detach()
        te = flat.reshape(B,N,C); del pv, vit, vf
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        sl = soft_prompt_embeds.shape[0]; sp = soft_prompt_embeds.to(device=ed, dtype=te.dtype)
        ce = torch.cat([sp.unsqueeze(0), te], 1); sm = torch.ones(1,sl,dtype=torch.long,device=ed)
        inp = {"inputs_embeds": ce, "attention_mask": torch.cat([sm, mask], 1)}
        if labels is not None:
            inp["labels"] = torch.cat([torch.full((1,sl),-100,dtype=labels.dtype,device=ed), labels.to(ed)], 1)
            apc = [p+sl for p in apo]
        oi = {"input_ids": ids, "attention_mask": mask, "_internvl_num_patches": np_, "_internvl_aspect_ratio": ar}
        return inp, {"orig_inputs": oi, "orig_input_ids": ids, "soft_len": sl,
                     "answer_token_ids": atids, "answer_pos_orig": apo, "answer_pos_combined": apc, "emb_device": ed}

    def build_inputs_with_soft_prompt(self, image, profile_text, soft_prompt_embeds, answer_text=None):
        if self.model_type == "internvl": return self._build_inputs_internvl(image, profile_text, soft_prompt_embeds, answer_text)
        instruction = INSTRUCTION
        msgs = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":f"{profile_text}\n\n{instruction}"}]}]
        pt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = pt if answer_text is None else pt+str(answer_text)
        oi = self.processor(text=full, images=image, return_tensors="pt")
        ids, mask = oi["input_ids"], oi["attention_mask"]
        labels, atids, apo, apc = None, None, None, None
        if answer_text is not None:
            atids = self.tokenizer.encode(str(answer_text), add_special_tokens=False)
            m = len(atids); Lf = ids.shape[1]; ast = Lf-m
            labels = torch.full_like(ids, -100); labels[0,ast:] = ids[0,ast:]; apo = list(range(ast, Lf))
        emb = self.model.get_input_embeddings(); ed = emb.weight.device
        ids, mask = ids.to(ed), mask.to(ed)
        if labels is not None: labels = labels.to(ed)
        for k, v in list(oi.items()):
            if torch.is_tensor(v): oi[k] = v.to(ed)
        te = emb(ids); sl = soft_prompt_embeds.shape[0]
        sp = soft_prompt_embeds.to(device=ed, dtype=te.dtype)
        ce = torch.cat([sp.unsqueeze(0), te], 1); sm = torch.ones(1,sl,dtype=torch.long,device=ed)
        inp = dict(oi); inp["attention_mask"] = torch.cat([sm, mask], 1); inp.pop("input_ids", None); inp["inputs_embeds"] = ce
        if labels is not None:
            inp["labels"] = torch.cat([torch.full((1,sl),-100,dtype=labels.dtype,device=ed), labels], 1); apc = [p+sl for p in apo]
        return inp, {"orig_inputs": oi, "orig_input_ids": ids, "soft_len": sl,
                     "answer_token_ids": atids, "answer_pos_orig": apo, "answer_pos_combined": apc, "emb_device": ed}

    def compute_losses(self, ma, gd, uids, lambda_attn_weight, br, loss_choice=None):
        g, a, cfg, eps = gd, ma, self.config, self.eps
        w = ((g + cfg.power_eps).pow(cfg.power_gamma))
        w = w / w.sum(1, keepdim=True)
        la_ = (w * g * (torch.log(g + eps) - torch.log(a + eps))).sum(1).mean()
        lr = self.prompt_basis.get_user_alpha_l2_loss(uids)
        if loss_choice is None:
            lc = torch.zeros((), device=la_.device, dtype=la_.dtype)
        else:
            lc = loss_choice
        # 与 train_att_attnlrp 一致使用 LM CE，但不乘 choice_loss_weight
        total = lc + lambda_attn_weight * la_ + br * lr
        return total, {k: float(v.detach().cpu()) for k, v in {
            "loss_choice": lc, "loss_attn": la_, "loss_reg": lr, "total_loss": total}.items()}

    def _optimizer_step(self, optimizer, uids):
        optimizer.step(); self.global_step += 1

    def _process_single_sample(self, i, imgs, pts, sps, cs, ama, choice_losses):
        img, pt, sp = imgs[i], pts[i], sps[i]
        ts = max(1, min(15, int(cs[i].item())+1)); at = slot_to_letter(ts)
        inp, meta = self.build_inputs_with_soft_prompt(img, pt, sp, at)
        ed = meta["emb_device"]; sp = sp.to(ed)
        with torch.enable_grad():
            out = self._model_forward(inp, output_attentions=True, use_cache=False, return_dict=True)
            if self.config.train_choice and getattr(out, "loss", None) is not None:
                choice_losses.append(out.loss)
            target, src = self._compute_target_scalar(out.logits, meta["answer_pos_combined"], meta["answer_token_ids"])
            E = self._attnlrp_backward(target, out.attentions, retain=True,
                                        create=self.config.attnlrp_create_graph,
                                        grad_scale=self.config.attnlrp_grad_scale)
            if not E:
                ma = sp.sum()*0.0 + torch.ones(NUM_SLOTS,device=sp.device,dtype=sp.dtype)/NUM_SLOTS
            else:
                R = self.attnlrp_propagation(E); src = max(0,min(int(src),R.shape[0]-1))
                vis = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])
                ma = self._resolve_slot_attention(R[src], vis, meta, img) if vis else torch.ones(NUM_SLOTS,device=DEVICE)/NUM_SLOTS
            ama.append(ma)

    def train_step(self, batch, optimizer, accumulation_steps=1, zero_grad=True, do_optimizer_step=True):
        uids, imgs, pts = batch["user_id"], batch["image"], batch["profile_text"]
        cs, gd = batch["choice_slot"].to(DEVICE), batch["gaze_dist"].to(DEVICE)
        bs = len(uids); sps = self.prompt_basis(uids); ama = []; cl = []
        for i in range(bs): self._process_single_sample(i, imgs, pts, sps, cs, ama, cl)
        ab = torch.stack(ama); gd = gd.to(ab.device)
        law = self.config.lambda_attn_weight
        loss_choice = torch.stack(cl).mean() if (self.config.train_choice and cl) else None
        total, ld = self.compute_losses(ab, gd, uids, law, self.config.beta_reg, loss_choice=loss_choice)
        if zero_grad: optimizer.zero_grad(set_to_none=True)
        (total/accumulation_steps).backward()
        if do_optimizer_step: self._optimizer_step(optimizer, uids)
        ld["lambda_attn_weight"] = float(law); return ld

    def train_step_batched(self, batch, optimizer, accumulation_steps=1, zero_grad=True, do_optimizer_step=True):
        uids, imgs, pts = batch["user_id"], batch["image"], batch["profile_text"]
        cs, gd = batch["choice_slot"].to(DEVICE), batch["gaze_dist"].to(DEVICE)
        bs = len(uids); sps = self.prompt_basis(uids)
        per_inp, per_meta = [], []
        for i in range(bs):
            ts = max(1,min(15,int(cs[i].item())+1)); at = slot_to_letter(ts)
            inp_i, meta_i = self.build_inputs_with_soft_prompt(imgs[i], pts[i], sps[i], at)
            per_inp.append(inp_i); per_meta.append(meta_i)
        if self.model_type != "internvl":
            ipad = self._find_special_id("<|image_pad|>")
            if ipad is None: return self.train_step(batch, optimizer, accumulation_steps, zero_grad, do_optimizer_step)
            for i in range(bs):
                pv = per_inp[i].pop("pixel_values",None); gthw = per_inp[i].pop("image_grid_thw",None)
                for k in ("pixel_values_videos","video_grid_thw"): per_inp[i].pop(k,None)
                if pv is not None:
                    vd = self.model.visual.get_dtype() if hasattr(self.model,"visual") and hasattr(self.model.visual,"get_dtype") else pv.dtype
                    with torch.no_grad():
                        ie = self.model.visual(pv.to(vd), grid_thw=gthw)
                        if isinstance(ie,(tuple,list)): ie=ie[0]
                        if hasattr(ie,"last_hidden_state"): ie=ie.last_hidden_state
                        if torch.is_tensor(ie) and ie.dim()==3 and ie.shape[0]==1: ie=ie[0]
                    oi = per_meta[i]["orig_input_ids"][0]; sl = per_meta[i]["soft_len"]
                    emb = per_inp[i]["inputs_embeds"]; mo = (oi==ipad)
                    mc = torch.zeros(emb.shape[1],dtype=torch.bool,device=mo.device); mc[sl:sl+mo.shape[0]]=mo
                    nt = min(int(mc.sum()), ie.shape[0])
                    if nt>0: emb[0, mc.nonzero(as_tuple=True)[0][:nt]] = ie[:nt].detach().to(emb.dtype)
                    del pv, ie
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        ed = per_meta[0]["emb_device"]; edt = per_inp[0]["inputs_embeds"].dtype; edim = per_inp[0]["inputs_embeds"].shape[-1]
        slens = [inp["inputs_embeds"].shape[1] for inp in per_inp]; ml = max(slens)
        be = torch.zeros(bs,ml,edim,device=ed,dtype=edt); bm = torch.zeros(bs,ml,dtype=torch.long,device=ed)
        bl = torch.full((bs,ml),-100,dtype=torch.long,device=ed)
        for i in range(bs):
            Li = slens[i]; be[i,:Li]=per_inp[i]["inputs_embeds"][0]; bm[i,:Li]=per_inp[i]["attention_mask"][0]
            if "labels" in per_inp[i]: bl[i,:Li]=per_inp[i]["labels"][0]
        bi = {"inputs_embeds":be,"attention_mask":bm}
        if self.config.train_choice: bi["labels"]=bl
        with torch.enable_grad():
            out = self._model_forward(bi, output_attentions=True, use_cache=False, return_dict=True)
            loss_choice = None
            if self.config.train_choice and getattr(out, "loss", None) is not None:
                loss_choice = out.loss
            tscalars, src_list = [], []
            for i in range(bs):
                m = per_meta[i]; sc, sp = self._compute_target_scalar(out.logits, m["answer_pos_combined"], m["answer_token_ids"], bi=i)
                tscalars.append(sc); src_list.append(sp)
            ss = sum(tscalars)
            L = len(out.attentions)
            lids = list(range(L))
            valid = [(l,out.attentions[l]) for l in lids if torch.is_tensor(out.attentions[l]) and out.attentions[l].requires_grad]
            all_grads = None
            if valid:
                _, va = zip(*valid)
                all_grads = torch.autograd.grad(ss, va, retain_graph=True, create_graph=self.config.attnlrp_create_graph, allow_unused=True)
            ama = []
            for i in range(bs):
                m = per_meta[i]; src = src_list[i]; sp = sps[i].to(ed); img = imgs[i]
                E = []
                if all_grads:
                    for (_,a),g in zip(valid, all_grads):
                        if g is None: continue
                        gl = g[i]*self.config.attnlrp_grad_scale if self.config.attnlrp_grad_scale!=1.0 else g[i]
                        E.append(self.attnlrp_layer_relevance(a[i], gl))
                if not E: ma = sp.sum()*0.0+torch.ones(NUM_SLOTS,device=sp.device,dtype=sp.dtype)/NUM_SLOTS
                else:
                    R = self.attnlrp_propagation(E); src = max(0,min(int(src),R.shape[0]-1))
                    vis = self.find_visual_token_indices(m["orig_input_ids"], m["soft_len"])
                    ma = self._resolve_slot_attention(R[src], vis, m, img) if vis else torch.ones(NUM_SLOTS,device=DEVICE)/NUM_SLOTS
                ama.append(ma)
        ab = torch.stack(ama); gd = gd.to(ab.device)
        law = self.config.lambda_attn_weight
        total, ld = self.compute_losses(ab, gd, uids, law, self.config.beta_reg, loss_choice=loss_choice)
        if zero_grad: optimizer.zero_grad(set_to_none=True)
        (total/accumulation_steps).backward()
        if do_optimizer_step: self._optimizer_step(optimizer, uids)
        ld["lambda_attn_weight"] = float(law); return ld

    def save_checkpoint(self, path):
        torch.save({'prompt_basis': self.prompt_basis.state_dict(), 'global_step': self.global_step}, path)
    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.prompt_basis.load_state_dict(ck['prompt_basis'], strict=False); self.global_step = ck.get('global_step',0)

# ═══════════════════════════════════════════════════════════════
#  Unified evaluation (supports baseline TF/FG + trained TF/FG)
# ═══════════════════════════════════════════════════════════════
def evaluate_freegen(trainer, samples, uf, pd, ed=None, use_soft_prompt=True):
    all_m, per_u, gt_l, gen_l = [], defaultdict(list), [], []
    ns, correct, r3, r5 = 0, 0, 0, 0
    for s in samples:
        uid, tid, ch, dw = s["user_id"], s["task_id"], s["choice_slot"], s["dwell_time"]
        img = _load_image(uid, tid, pd, ed)
        if img is None: ns+=1; continue
        prof = uf.get(uid,{}); pt = f"Top_genre: {prof.get('Top_genre','')}, Preferred_genres: {prof.get('Preferred_genres','')}"
        gt = slot_to_letter(ch)
        try:
            if use_soft_prompt: attn, gen = compute_freegeneration_attention(trainer, uid, img, pt)
            else: attn, gen = compute_baseline_freegeneration_attention(trainer, img, pt)
        except Exception:
            ns += 1
            continue
        if gen.strip().upper()==gt.strip().upper(): correct+=1
        lp = None
        try:
            olp = compute_option_logprobs(trainer, img, pt, user_id=uid if use_soft_prompt else None)
            rk = np.argsort(olp)[::-1]; gi = ch-1
            if gi in rk[:3]: r3+=1
            if gi in rk[:5]: r5+=1
            lp = np.exp(olp); lp = lp/lp.sum()
        except: pass
        gd = gaze_distribution_from_dwell_time(dw)
        m = compute_sample_metrics(attn, gd, ch-1, logit_probs=lp)
        all_m.append(m); per_u[uid].append(m)
        gt_l.append(gt.strip().upper()); gen_l.append(_normalize_option_letter(gen))
    if not all_m: return {}
    r = {}
    for n in all_m[0]:
        r[f"micro_{n}"] = float(np.mean([m[n] for m in all_m]))
        r[f"macro_{n}"] = float(np.mean([np.mean([m[n] for m in ms]) for ms in per_u.values()]))
    r.update(n_samples=len(all_m), n_users=len(per_u), answer_accuracy=correct/len(all_m),
             **{f"recall@{k}":v/len(all_m) for k,v in [(3,r3),(5,r5)]})
    pfx = "tr_fg" if use_soft_prompt else "bl_fg"
    for k,v in compute_option_ratios(gen_l).items(): r[f"{pfx}_ratio_{k}"]=float(v)
    if not use_soft_prompt:
        for k,v in compute_option_ratios(gt_l).items(): r[f"gt_ratio_{k}"]=float(v)
    return r

def evaluate_tf(trainer, samples, uf, pd, ed=None, use_soft_prompt=True):
    """Teacher-forcing evaluation (baseline or trained)."""
    all_m, per_u = [], defaultdict(list)
    ns, r3, r5 = 0, 0, 0
    for s in samples:
        uid, tid, ch, dw = s["user_id"], s["task_id"], s["choice_slot"], s["dwell_time"]
        img = _load_image(uid, tid, pd, ed)
        if img is None: ns+=1; continue
        prof = uf.get(uid,{}); pt = f"Top_genre: {prof.get('Top_genre','')}, Preferred_genres: {prof.get('Preferred_genres','')}"
        at = slot_to_letter(ch)
        try:
            if use_soft_prompt:
                ma = trainer.compute_model_attention_single(uid, img, pt, at)
            else:
                ma = compute_baseline_attnlrp_attention(trainer, img, pt, at)
            if torch.is_tensor(ma): ma = ma.detach().float().cpu().numpy()
        except Exception:
            ns += 1
            continue
        lp = None
        try:
            olp = compute_option_logprobs(trainer, img, pt, user_id=uid if use_soft_prompt else None)
            rk = np.argsort(olp)[::-1]; gi = ch-1
            if gi in rk[:3]: r3+=1
            if gi in rk[:5]: r5+=1
            lp = np.exp(olp); lp = lp/lp.sum()
        except: pass
        gd = gaze_distribution_from_dwell_time(dw)
        m = compute_sample_metrics(ma, gd, ch-1, logit_probs=lp)
        all_m.append(m); per_u[uid].append(m)
    if not all_m: return {}
    r = {}
    for n in all_m[0]:
        r[f"micro_{n}"] = float(np.mean([m[n] for m in all_m]))
        r[f"macro_{n}"] = float(np.mean([np.mean([m[n] for m in ms]) for ms in per_u.values()]))
    r.update(n_samples=len(all_m), n_users=len(per_u), **{f"recall@{k}":v/len(all_m) for k,v in [(3,r3),(5,r5)]})
    return r

# ═══════════════════════════════════════════════════════════════
#  Pipeline: fold / CV / grid search / final / multi-seed
# ═══════════════════════════════════════════════════════════════
def _run_epoch(trainer, loader, optimizer):
    losses, accum, nb = [], GRADIENT_ACCUMULATION_STEPS, len(loader)
    fn = trainer.train_step_batched if USE_BATCHED_FORWARD else trainer.train_step
    for bi, batch in enumerate(loader):
        ld = fn(batch, optimizer, accumulation_steps=accum,
                zero_grad=(bi%accum==0), do_optimizer_step=((bi+1)%accum==0 or bi+1==nb))
        losses.append(ld)
    return {k: float(np.mean([d.get(k,0) for d in losses])) for k in losses[0]}

def _split_leave1_test_then_kfold(samples, K=5, seed=42):
    rng = np.random.RandomState(seed)
    by_user = defaultdict(list)
    for i, s in enumerate(samples): by_user[s["user_id"]].append(i)
    test_set, hist_by_user, always_train = set(), {}, set()
    for uid, idxs in by_user.items():
        arr = np.array(idxs); rng.shuffle(arr)
        if len(arr)==1: always_train.add(int(arr[0])); continue
        test_set.add(int(arr[0])); hist_by_user[uid] = [int(x) for x in arr[1:]]
    fold_val = [set() for _ in range(K)]
    for uid, hidxs in hist_by_user.items():
        arr = np.array(hidxs); rng.shuffle(arr)
        if len(arr)==1: always_train.add(int(arr[0])); continue
        off = rng.randint(0, K)
        for j, ix in enumerate(arr): fold_val[(off+j)%K].add(int(ix))
    hist_all = set()
    for h in hist_by_user.values(): hist_all.update(h)
    pool = hist_all | always_train
    return list(test_set), [(list((pool-v)|always_train), list(v)) for v in fold_val]

def train_one_fold(trainer, train_s, val_s, uf, pd, ed, hp, ne, ee):
    apply_hyperparams(trainer, hp); reset_prompt_basis(trainer, hp)
    for uid in set(s["user_id"] for s in train_s+val_s): trainer.prompt_basis.get_or_create_user_alpha(uid)
    opt = build_optimizer(trainer, hp)
    loader = DataLoader(GazeDataset(train_s, uf, pd, expanded_dirs=ed), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    best_sc = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf"); best_m, best_ep, best_st = None, 0, None
    for epoch in range(1, ne+1):
        avg = _run_epoch(trainer, loader, opt)
        if (epoch%ee==0) or epoch==ne:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            vm = evaluate_tf(trainer, val_s, uf, pd, ed)
            sc, _ = compute_primary_score(vm)
            improved = np.isfinite(sc) and ((sc<best_sc) if PRIMARY_LOWER_IS_BETTER else (sc>best_sc))
            if improved:
                best_sc, best_m, best_ep = sc, vm, epoch
                best_st = {k: v.cpu().clone() for k, v in trainer.prompt_basis.state_dict().items()}
    return {"best_epoch": best_ep, "best_val_score": best_sc, "best_val_metrics": best_m or {}, "best_state": best_st}

def cross_validate(trainer, samples, folds, uf, pd, ed, hp, ne, ee):
    K = len(folds); results = []
    for f in range(K):
        ti, vi = folds[f]
        r = train_one_fold(trainer, [samples[i] for i in ti], [samples[i] for i in vi], uf, pd, ed, hp, ne, ee)
        results.append(r)
    valid = [r for r in results if r["best_val_metrics"]]
    if not valid: return {"avg_metrics":{}, "avg_best_epoch":0, "fold_results":results}
    avg_m = {k: float(np.mean([r["best_val_metrics"][k] for r in valid])) for k in valid[0]["best_val_metrics"]}
    return {"avg_metrics":avg_m, "avg_primary_score":compute_primary_score(avg_m)[0],
            "avg_best_epoch":float(np.mean([r["best_epoch"] for r in valid])), "fold_results":results}

def grid_search_cv(trainer, samples, folds, uf, pd, ed, param_grid, ne, ee):
    keys = sorted(param_grid.keys()); combos = list(product(*(param_grid[k] for k in keys)))
    best_sc = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf"); best_hp, best_cv = None, None
    for combo in combos:
        hp = dict(DEFAULT_HYPERPARAMS); hp.update(zip(keys, combo))
        cv = cross_validate(trainer, samples, folds, uf, pd, ed, hp, ne, ee)
        sc = cv.get("avg_primary_score", float("nan"))
        if np.isfinite(sc) and ((sc<best_sc) if PRIMARY_LOWER_IS_BETTER else (sc>best_sc)):
            best_sc, best_hp, best_cv = sc, hp.copy(), cv
    with open(os.path.join(OUTPUT_DIR, f"attnlrp_grid_{RUN_TIMESTAMP}.json"), "w") as f: json.dump({"best_hp": best_hp, "best_score": best_sc}, f, indent=2)
    return best_hp, best_cv

def final_train_and_test(trainer, all_samples, test_idx, uf, pd, ed, hp, ne, val_ratio=0.0):
    test_set = set(test_idx); train_all = [s for i,s in enumerate(all_samples) if i not in test_set]
    test_s = [all_samples[i] for i in test_idx]
    use_val = val_ratio>0 and len(train_all)>=10
    if use_val:
        rng = np.random.RandomState(SEED); nv = max(1,int(len(train_all)*val_ratio)); perm = rng.permutation(len(train_all))
        vi = set(perm[:nv].tolist()); train_s = [s for i,s in enumerate(train_all) if i not in vi]; val_s = [train_all[i] for i in sorted(vi)]
    else: train_s, val_s = train_all, []
    apply_hyperparams(trainer, hp); reset_prompt_basis(trainer, hp)
    for uid in set(s["user_id"] for s in all_samples): trainer.prompt_basis.get_or_create_user_alpha(uid)
    opt = build_optimizer(trainer, hp)
    loader = DataLoader(GazeDataset(train_s, uf, pd, expanded_dirs=ed), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    best_sc = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf"); best_ep, best_st, no_imp = 0, None, 0
    for epoch in range(1, ne+1):
        avg = _run_epoch(trainer, loader, opt)
        do_eval = use_val and ((epoch%EVAL_EVERY_N_EPOCHS==0) or epoch==ne)
        if do_eval:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            vm = evaluate_tf(trainer, val_s, uf, pd, ed)
            sc, _ = compute_primary_score(vm)
            improved = np.isfinite(sc) and ((sc<best_sc) if PRIMARY_LOWER_IS_BETTER else (sc>best_sc))
            if improved: best_sc, best_ep = sc, epoch; best_st = {k:v.cpu().clone() for k,v in trainer.prompt_basis.state_dict().items()}; no_imp=0
            else: no_imp+=1
            if no_imp>=5: break
    if use_val and best_st: trainer.prompt_basis.load_state_dict(best_st)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    trainer.save_checkpoint(os.path.join(CHECKPOINT_DIR, f"attnlrp_final_{RUN_TIMESTAMP}.pt"))
    bl_tf = evaluate_tf(trainer, test_s, uf, pd, ed, use_soft_prompt=False)
    bl_fg = evaluate_freegen(trainer, test_s, uf, pd, ed, use_soft_prompt=False)
    tr_tf = evaluate_tf(trainer, test_s, uf, pd, ed, use_soft_prompt=True)
    tr_fg = evaluate_freegen(trainer, test_s, uf, pd, ed, use_soft_prompt=True)
    with open(os.path.join(OUTPUT_DIR, f"att_attnlrp_final_{RUN_TIMESTAMP}.json"), "w") as f:
        json.dump({"bl_tf":bl_tf,"bl_fg":bl_fg,"tr_tf":tr_tf,"tr_fg":tr_fg,"hp":hp}, f, indent=2, ensure_ascii=False)
    combined = dict(tr_tf)
    if tr_fg: combined["answer_accuracy"] = tr_fg.get("answer_accuracy", float("nan"))
    combined["_four_way"] = {"bl_tf":bl_tf,"bl_fg":bl_fg,"tr_tf":tr_tf,"tr_fg":tr_fg}
    return combined

def multi_seed_final(trainer, all_samples, test_idx, uf, pd, ed, hp, ne, val_ratio=0.0):
    seeds = MULTI_SEEDS
    all_m, all_fw = [], []
    best_sc = float("inf") if PRIMARY_LOWER_IS_BETTER else float("-inf"); best_seed = None
    for seed in seeds:
        set_all_seeds(seed)
        m = final_train_and_test(trainer, all_samples, test_idx, uf, pd, ed, hp, ne, val_ratio=val_ratio)
        all_m.append((seed, m)); fw = m.pop("_four_way", None)
        if fw: all_fw.append((seed, fw))
        sc, _ = compute_primary_score(m)
        if np.isfinite(sc) and ((sc<best_sc) if PRIMARY_LOWER_IS_BETTER else (sc>best_sc)): best_sc, best_seed = sc, seed
    avg = {}; all_keys = set(); [all_keys.update(m.keys()) for _,m in all_m]
    for k in sorted(all_keys):
        vals = [m[k] for _,m in all_m if k in m and isinstance(m[k],(int,float)) and np.isfinite(m[k])]
        if vals: avg[k] = float(np.mean(vals))
    return avg, all_fw

# ═══════════════════════════════════════════════════════════════
def main():
    uf = load_user_features(USER_FEATURES_CSV)
    ml = load_movie_layout_from_item_features(ITEM_FEATURES_CSV, 1, 35, 15)
    samples = load_gaze_data(GAZE_DATA_CSV, ml)
    expanded_dirs = EXPANDED_DIRS
    samples = [s for s in samples if _load_image(s["user_id"], s["task_id"], POSTER_IMAGES_DIR, expanded_dirs)]
    test_idx, folds = _split_leave1_test_then_kfold(samples, K_FOLDS, SEED)
    trainer = AttnLRPTrainer(MODEL_NAME, MODEL_TYPE)
    if RUN_GRID_SEARCH:
        best_hp, best_cv = grid_search_cv(trainer, samples, folds, uf, POSTER_IMAGES_DIR, expanded_dirs, PARAM_GRID, NUM_EPOCHS_CV, EVAL_EVERY_N_EPOCHS)
        final_ep = max(1, round(best_cv.get("avg_best_epoch", NUM_EPOCHS_FINAL)))
    else:
        best_hp = dict(DEFAULT_HYPERPARAMS); final_ep = NUM_EPOCHS_FINAL
    fvr = NO_CV_VAL_RATIO if not RUN_GRID_SEARCH else 0.0
    if USE_MULTI_SEED:
        test_m, all_fw = multi_seed_final(trainer, samples, test_idx, uf, POSTER_IMAGES_DIR, expanded_dirs, best_hp, final_ep, fvr)
    else:
        set_all_seeds(SEED)
        test_m = final_train_and_test(trainer, samples, test_idx, uf, POSTER_IMAGES_DIR, expanded_dirs, best_hp, final_ep, fvr)

if __name__ == "__main__":
    main()