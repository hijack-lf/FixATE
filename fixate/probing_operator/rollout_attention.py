"""Rollout attention mixin, baseline/freegen, data loading."""
import os, sys, math, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from config.rollout_config import *

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Expanded UI images (filenames include User / TaskID)
EXPANDED_DIRS = [(1, 35, os.path.join(_REPO, "attention_heatmaps", "Expanded_dataset_index"))]

# ── InternVL preprocessing (same as attnlrp) ──
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
def _build_ivl_transform(sz=448):
    return T.Compose([T.Lambda(lambda i: i.convert('RGB') if i.mode!='RGB' else i),
                      T.Resize((sz,sz), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def _preprocess_image_for_internvl(image, input_size=448, max_num=12):
    w, h = image.size; ar = w/h
    ratios = sorted(set((i,j) for n in range(1,max_num+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max_num), key=lambda x:x[0]*x[1])
    best = min(ratios, key=lambda r: abs(ar - r[0]/r[1]))
    tw, th = input_size*best[0], input_size*best[1]
    ri = image.resize((tw,th)); tiles = []
    for idx in range(best[0]*best[1]):
        c, r = idx%(tw//input_size), idx//(tw//input_size)
        tiles.append(ri.crop((c*input_size, r*input_size, (c+1)*input_size, (r+1)*input_size)))
    if len(tiles) != 1: tiles.append(image.resize((input_size, input_size)))
    tf = _build_ivl_transform(input_size)
    return torch.stack([tf(t) for t in tiles]), len(tiles), best

# ── Data utilities ──
def slot_to_letter(s): return chr(ord('A') + max(1, min(15, s)) - 1)
def _normalize_option_letter(s):
    if not s: return "OTHER"
    u = s.strip().upper(); return u if len(u)==1 and 'A'<=u<='O' else "OTHER"
def compute_option_ratios(letters, ns=15):
    from collections import Counter
    n = len(letters)
    if n == 0: return {}
    cnt = Counter(letters)
    return {chr(ord('A')+i): cnt.get(chr(ord('A')+i),0)/n for i in range(ns)} | {"OTHER": cnt.get("OTHER",0)/n}
def gaze_distribution_from_dwell_time(dw, tau=GAZE_TAU, eps=1e-8):
    t = dw.astype(np.float32); g = t/(t.sum()+eps)
    g = np.exp(np.log(g+eps)/tau); return g/(g.sum()+eps)

def load_user_features(path):
    uf = {}
    with open(path,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            uid = r.get('UserID','').strip()
            if uid: uf[uid] = {'Top_genre': r.get('Top_genre','').strip(), 'Preferred_genres': r.get('Preferred_genres','').strip()}
    return uf

def load_movie_layout_from_item_features(path, task_id_min=1, task_id_max=35, max_movie_pos=15):
    layout = {}
    with open(path,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                tid = int(r.get('TaskID','')); cp = int(r.get('Carousel_position',''))
                mp = int(r.get('Movie_position_in_carousel','')); mid = int(r.get('MovieID',''))
                if task_id_min<=tid<=task_id_max and 1<=mp<=max_movie_pos and 1<=cp<=3:
                    layout[(tid,cp,mp)] = mid
            except: continue
    return layout

def load_gaze_data(path, movie_layout):
    grouped = defaultdict(lambda: {'fixations': defaultdict(float), 'clicks': set()})
    with open(path,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            uid, tids = r.get('UserID','').strip(), r.get('TaskID','').strip()
            if not uid or not tids: continue
            try: tid = int(tids)
            except: continue
            if not 1<=tid<=35: continue
            key = (uid, tid)
            fd = r.get('Fixation_Duration','').strip()
            if fd:
                try:
                    dur = float(fd); cp = int(float(r.get('Fixation_AOI_Carousel_position','').strip()))
                    mp = int(float(r.get('Fixation_AOI_Movie_position_in_carousel','').strip()))
                    if 1<=cp<=3 and 1<=mp<=5: grouped[key]['fixations'][(cp-1)*5+mp] += dur
                except: pass
            cmid = r.get('Click_AOI_MovieID','').strip()
            if cmid:
                try: grouped[key]['clicks'].add(int(float(cmid)))
                except: pass
    samples = []
    for (uid, tid), data in grouped.items():
        dw = np.zeros(NUM_SLOTS, dtype=np.float32)
        for sid, dur in data['fixations'].items():
            if 1<=sid<=NUM_SLOTS: dw[sid-1] = dur
        if dw.sum() <= 0: continue
        cs = None
        for cmid in data['clicks']:
            for rn in range(1,4):
                for cn in range(1,16):
                    if movie_layout.get((tid,rn,cn)) == cmid:
                        cs = (rn-1)*NUM_COLS + ((cn-1)%NUM_COLS) + 1; break
                if cs: break
            if cs: break
        if cs: samples.append({'user_id': uid, 'task_id': tid, 'choice_slot': cs, 'dwell_time': dw})
    return samples

def _load_image(uid, tid, poster_dir, expanded_dirs=None):
    if expanded_dirs:
        for tmin, tmax, dp in expanded_dirs:
            if tmin<=tid<=tmax:
                p = os.path.join(dp, f"User_{uid}_TaskID_{tid:02d}_final_interface.png")
                if os.path.exists(p): return Image.open(p).convert("RGB")
    p = os.path.join(poster_dir, f"TaskID_{tid:02d}_posters.png")
    return Image.open(p).convert("RGB") if os.path.exists(p) else None

class GazeDataset(Dataset):
    def __init__(self, samples, user_features, poster_dir, expanded_dirs=None):
        self.samples, self.uf, self.pd, self.ed = samples, user_features, poster_dir, expanded_dirs or []
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]; uid, tid = s['user_id'], s['task_id']
        ip = None
        for tmin, tmax, dp in self.ed:
            if tmin<=tid<=tmax: ip = os.path.join(dp, f"User_{uid}_TaskID_{tid:02d}_final_interface.png"); break
        if ip is None: ip = os.path.join(self.pd, f"TaskID_{tid:02d}_posters.png")
        image = Image.open(ip).convert("RGB")
        prof = self.uf.get(uid, {}); pt = f"Top_genre: {prof.get('Top_genre','')}, Preferred_genres: {prof.get('Preferred_genres','')}"
        return {'user_id': uid, 'task_id': tid, 'image': image, 'profile_text': pt,
                'choice_slot': s['choice_slot']-1, 'gaze_dist': torch.from_numpy(gaze_distribution_from_dwell_time(s['dwell_time']))}

class PromptBasisModule(nn.Module):
    def __init__(self, num_basis=NUM_BASIS, num_soft_tokens=NUM_SOFT_TOKENS, hidden_dim=3584, use_user_alpha=USE_USER_ALPHA):
        super().__init__()
        self.num_basis, self.num_soft_tokens, self.hidden_dim, self.use_user_alpha = num_basis, num_soft_tokens, hidden_dim, use_user_alpha
        self.basis = nn.Parameter(torch.randn(num_basis, num_soft_tokens, hidden_dim)*0.01)
        self.user_alphas = nn.ParameterDict()
    def get_or_create_user_alpha(self, uid):
        if uid not in self.user_alphas: self.user_alphas[uid] = nn.Parameter(torch.zeros(self.num_basis, device=self.basis.device))
        return self.user_alphas[uid]
    def forward(self, uids):
        out = []
        for uid in uids:
            if self.use_user_alpha:
                a = self.get_or_create_user_alpha(uid).to(self.basis.device)
                pi = F.softmax(a, dim=0).to(dtype=self.basis.dtype)
            else: pi = torch.ones(self.num_basis, device=self.basis.device, dtype=self.basis.dtype)/self.num_basis
            out.append(torch.einsum('b,bmd->md', pi, self.basis))
        return torch.stack(out)
    def get_user_alpha_l2_loss(self, uids):
        if not self.use_user_alpha: return torch.zeros((), device=self.basis.device)
        return sum(torch.sum(self.get_or_create_user_alpha(u)**2) for u in uids)/len(uids)


# ── Rollout Attention Mixin ──

class RolloutAttentionMixin:
    """Mixin: Attention Rollout computation + patch→slot aggregation."""

    def _find_special_id(self, tok):
        try:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            return None if tid is None or tid == self.tokenizer.unk_token_id else tid
        except: return None

    def find_visual_token_indices(self, input_ids, soft_len):
        ids = input_ids[0].tolist(); vis = []
        vs, ve = self._find_special_id("<|vision_start|>"), self._find_special_id("<|vision_end|>")
        if vs and ve and vs in ids and ve in ids:
            s, e = ids.index(vs), ids.index(ve)
            if e > s+1: vis = list(range(s+1, e))
        for tok in ["<|image_pad|>", "<IMG_CONTEXT>", "<image>"]:
            if vis: break
            tid = self._find_special_id(tok)
            if tid: vis = [i for i, t in enumerate(ids) if t == tid]
        return [i + soft_len for i in vis]

    def _compute_rollout(self, attentions):
        """Compute rollout: mean over heads → 0.5*A+0.5*I → row-normalize → multiply layers (fixed, no discard)."""
        rollout = None
        for la in attentions:
            if la.dim() == 4:
                attn = la[0].mean(dim=0).to(torch.float32)
            elif la.dim() == 3:
                attn = la.mean(dim=0).to(torch.float32)
            else:
                raise ValueError(f"attention tensor must be 3D or 4D, got shape {tuple(la.shape)}")
            N = attn.shape[0]; ah = 0.5*attn + 0.5*torch.eye(N, device=attn.device, dtype=attn.dtype)
            ah = ah / (ah.sum(-1, keepdim=True) + 1e-12)
            rollout = ah if rollout is None else ah @ rollout
        return rollout

    def aggregate_patch_saliency_to_slots_torch(self, ps, image_size, pg):
        dev, dt = ps.device, ps.dtype; Hp, Wp = pg
        if ps.dim() == 1:
            n, hw = ps.numel(), Hp*Wp
            if n == hw: ps2 = ps.view(Hp, Wp)
            elif hw == 4*n: ps2 = ps.view(Hp//2, Wp//2).repeat_interleave(2,0).repeat_interleave(2,1)[:Hp,:Wp]
            elif n > hw: ps2 = ps[:hw].view(Hp, Wp)
            else: pad = torch.zeros(hw, device=dev, dtype=dt); pad[:n] = ps; ps2 = pad.view(Hp, Wp)
        else: ps2 = ps; Hp, Wp = ps2.shape
        ps2 = torch.relu(ps2)
        iw, ih = float(image_size[0]), float(image_size[1])
        gi, gj = torch.meshgrid(torch.arange(Hp, device=dev), torch.arange(Wp, device=dev), indexing="ij")
        cx, cy = (gj.float()+0.5)*iw/Wp, (gi.float()+0.5)*ih/Hp
        xs, xe, ys, ye = iw*0.05, iw*0.95, ih*0.1, ih*0.95
        mask = (cx>=xs)&(cx<xe)&(cy>=ys)&(cy<ye)
        col = torch.clamp(((cx-xs)/((xe-xs)/NUM_COLS)).long(), 0, NUM_COLS-1)
        row = torch.clamp(((cy-ys)/((ye-ys)/NUM_ROWS)).long(), 0, NUM_ROWS-1)
        idx = (row*NUM_COLS+col).reshape(-1); vals = ps2.reshape(-1)[mask.reshape(-1)]
        out = torch.zeros(NUM_SLOTS, device=dev, dtype=dt)
        if vals.numel()>0: out = out.scatter_add(0, idx[mask.reshape(-1)], vals)
        return out/(out.sum()+1e-8)

    def _get_ivl_patch_grid(self, ar, n): t = int(math.sqrt(self._internvl_num_image_token)); return (t*ar[0], t*ar[1])
    def _get_ivl_content_count(self, ar): t = int(math.sqrt(self._internvl_num_image_token)); return t*ar[0]*t*ar[1]
    def _fix_patch_grid(self, pg, nv):
        H, W = pg; hw = H*W
        if hw == nv: return pg
        if hw == 4*nv and H%2==0 and W%2==0: return (H//2, W//2)
        if nv == 4*hw: return (H*2, W*2)
        s = int(math.sqrt(nv))
        for h in range(s, 0, -1):
            if nv%h==0: return (h, nv//h)
        return (1, nv)
    def _infer_patch_grid(self, oi, nv):
        if self.model_type == "internvl":
            ar = oi.get("_internvl_aspect_ratio")
            if ar: return self._get_ivl_patch_grid(ar, nv)
        for key in ("image_grid_thw", "grid_thw", "vision_grid_thw"):
            v = oi.get(key)
            if v is not None and torch.is_tensor(v):
                thw = v.reshape(-1)
                if thw.numel()>=3: return (int(thw[-2].item()), int(thw[-1].item()))
        s = int(math.sqrt(nv)); return (max(1,s), max(1,nv//max(1,s)))

    def _rollout_to_slot_attention(self, rollout, query_pos, vis_indices, meta, image):
        """From rollout matrix → slot attention using query_pos row."""
        if rollout is None or not vis_indices:
            return torch.ones(NUM_SLOTS, device=rollout.device if rollout is not None else DEVICE, dtype=torch.float32)/NUM_SLOTS
        N = rollout.shape[0]; qp = max(0, min(query_pos, N-1))
        qattn = rollout[qp]
        vis_indices = [v for v in vis_indices if 0<=v<N]
        if not vis_indices: return torch.ones(NUM_SLOTS, device=rollout.device)/NUM_SLOTS
        oi = meta["orig_inputs"]
        if self.model_type == "internvl":
            ar = oi.get("_internvl_aspect_ratio")
            if ar:
                cn = self._get_ivl_content_count(ar)
                if len(vis_indices) > cn: vis_indices = vis_indices[:cn]
        vr = torch.relu(qattn[vis_indices])
        pg = self._fix_patch_grid(self._infer_patch_grid(oi, len(vis_indices)), len(vis_indices))
        return self.aggregate_patch_saliency_to_slots_torch(vr, (image.width, image.height), pg)

    def compute_model_attention_single(self, user_id, image, profile_text, answer_text=None):
        """Rollout-based single sample eval attention (with soft prompt)."""
        sp = self.prompt_basis([user_id])[0]
        inp, meta = self.build_inputs_with_soft_prompt(image=image, profile_text=profile_text,
                                                       soft_prompt_embeds=sp, answer_text=answer_text)
        with torch.no_grad():
            out = self._model_forward(inp, output_attentions=True, use_cache=False, return_dict=True)
        if not out.attentions: return torch.ones(NUM_SLOTS, device=DEVICE)/NUM_SLOTS
        rollout = self._compute_rollout([a[0:1] for a in out.attentions])
        # Find answer position for query
        src_pos = rollout.shape[0]-1
        if answer_text:
            atids = self.tokenizer.encode(answer_text, add_special_tokens=False)
            ids_list = meta["orig_input_ids"][0].tolist(); sl = meta["soft_len"]
            for tid in atids:
                for pos in range(len(ids_list)-1, -1, -1):
                    if ids_list[pos] == tid:
                        prev = pos + sl - 1
                        if prev >= 0: src_pos = min(prev, rollout.shape[0]-1)
                        break
        vis = self.find_visual_token_indices(meta["orig_input_ids"], meta["soft_len"])
        attn = self._rollout_to_slot_attention(rollout, src_pos, vis, meta, image).detach().float()
        s = attn.sum(); return attn/s if s > 1e-9 else torch.ones(NUM_SLOTS, dtype=torch.float32)/NUM_SLOTS


# ── Standalone: answer generation + baseline/freegen ──
class _AnswerConstraint:
    def __init__(self, ids): self.ids = ids
    def __call__(self, input_ids, scores):
        m = torch.full_like(scores, float('-inf')); m[:, self.ids] = 0; return scores+m

def _generate_answer(trainer, inputs, is_ivl=False):
    safe = {k: v for k, v in inputs.items() if not k.startswith("_")}
    ilen = safe.get("inputs_embeds", safe.get("input_ids", torch.empty(1,0))).shape[1]
    kw = dict(**safe, max_new_tokens=1, do_sample=False, pad_token_id=trainer.tokenizer.pad_token_id,
              logits_processor=[_AnswerConstraint(trainer._answer_token_ids)])
    out = (trainer.model.language_model if is_ivl else trainer.model).generate(**kw)
    tok = out[0][ilen:] if out.shape[1]>ilen else out[0]
    return trainer.tokenizer.decode(tok, skip_special_tokens=True).strip()

def _build_ivl_embeds(trainer, image, profile_text, answer_text=None):
    dev = next(trainer.model.parameters()).device; emb = trainer.model.get_input_embeddings()
    txt, pv_cpu, np_, ar = trainer._build_internvl_text(profile_text, image, answer_text)
    tok = trainer.tokenizer(txt, return_tensors="pt")
    ids, mask = tok["input_ids"].to(dev), tok["attention_mask"].to(dev)
    te = emb(ids).clone(); pv = pv_cpu.to(device=dev, dtype=te.dtype)
    with torch.no_grad(): vit = trainer.model.extract_feature(pv)
    B,N,C = te.shape; flat = te.reshape(B*N,C)
    sel = (ids.reshape(B*N)==trainer.model.img_context_token_id)
    vf = vit.reshape(-1,C).to(flat.device); nt = min(int(sel.sum()), vf.shape[0])
    if nt>0: flat[sel.nonzero(as_tuple=True)[0][:nt]] = vf[:nt].detach()
    te = flat.reshape(B,N,C); del pv, vit, vf
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return te, ids, mask, ar

def _build_baseline_inputs(trainer, image, profile_text, answer_text=None, for_generate=False):
    dev = next(trainer.model.parameters()).device
    if trainer.model_type == "internvl":
        ans = None if for_generate else answer_text
        te, ids, mask, ar = _build_ivl_embeds(trainer, image, profile_text, ans)
        return {"inputs_embeds": te, "attention_mask": mask, "_internvl_aspect_ratio": ar}, ids, {}
    pfx = f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{profile_text}<|im_end|>\n<|im_start|>assistant\n"
    text = pfx if for_generate else pfx+str(answer_text)
    inp = trainer.processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inp = {k: v.to(dev) for k, v in inp.items()}
    return inp, inp["input_ids"], {}

def _rollout_from_inputs(trainer, inputs, input_ids, answer_text, image, soft_len=0):
    """Core rollout extraction (no soft prompt)."""
    uniform = np.ones(NUM_SLOTS, dtype=np.float32)/NUM_SLOTS
    with torch.no_grad():
        out = trainer._model_forward(inputs, output_attentions=True, use_cache=False, return_dict=True)
    if not out.attentions: return uniform
    rollout = trainer._compute_rollout([a[0:1] for a in out.attentions])
    if rollout is None: return uniform
    src_pos = rollout.shape[0]-1
    if answer_text:
        atids = trainer.tokenizer.encode(answer_text, add_special_tokens=False)
        ids_list = input_ids[0].tolist()
        for tid in atids:
            for pos in range(len(ids_list)-1,-1,-1):
                if ids_list[pos]==tid:
                    prev = pos+soft_len-1
                    if prev>=0: src_pos = min(prev, rollout.shape[0]-1)
                    break
    vis = []
    ids_ = input_ids[0].tolist()
    if trainer.model_type == "internvl":
        cid = trainer.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if cid: vis = [i for i, t in enumerate(ids_) if t == cid]
    if not vis:
        vs = trainer.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve = trainer.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        try: s, e = ids_.index(vs), ids_.index(ve, ids_.index(vs)); vis = list(range(s+1, e))
        except: pass
    if not vis: return uniform
    vis_s = [i+soft_len for i in vis]; N = rollout.shape[0]
    vis_s = [v for v in vis_s if 0<=v<N]
    if trainer.model_type == "internvl":
        ar = inputs.get("_internvl_aspect_ratio")
        if ar:
            cn = trainer._get_ivl_content_count(ar)
            if len(vis_s)>cn: vis_s = vis_s[:cn]
    vr = torch.relu(rollout[max(0,min(src_pos,N-1))][vis_s]).float()
    pg = trainer._fix_patch_grid(trainer._infer_patch_grid(inputs, len(vis_s)), len(vis_s))
    sa = trainer.aggregate_patch_saliency_to_slots_torch(vr, (image.width, image.height), pg).detach().cpu().numpy()
    s = sa.sum(); return sa/s if s>1e-9 else uniform

def compute_baseline_rollout_attention(trainer, image, profile_text, answer_text):
    trainer.model.eval()
    mi, ids, _ = _build_baseline_inputs(trainer, image, profile_text, answer_text)
    return _rollout_from_inputs(trainer, mi, ids, answer_text, image)

def compute_rollout_freegeneration_with_soft_prompt(trainer, user_id, image, profile_text):
    """Trained FG: soft prompt → generate → rollout."""
    trainer.model.eval(); ivl = trainer.model_type == "internvl"
    ed = trainer.model.get_input_embeddings().weight.device
    if ivl:
        sp = trainer.prompt_basis([user_id])[0]
        ig, _ = trainer.build_inputs_with_soft_prompt(image=image, profile_text=profile_text, soft_prompt_embeds=sp, answer_text=None)
        gi = {k: v for k, v in ig.items() if k != "labels"}
    else:
        dev = next(trainer.model.parameters()).device
        text = f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{profile_text}<|im_end|>\n<|im_start|>assistant\n"
        inp = trainer.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        sp = trainer.prompt_basis([user_id])[0]; emb = trainer.model.get_input_embeddings()
        te = emb(inp["input_ids"].to(ed)); spe = sp.to(device=ed, dtype=te.dtype)
        ge = torch.cat([spe.unsqueeze(0), te], 1); sl = spe.shape[0]
        gi = {k: v for k, v in inp.items() if k != "input_ids"}
        gi["inputs_embeds"] = ge; gi["attention_mask"] = torch.cat([torch.ones(1,sl,dtype=torch.long,device=ed), inp["attention_mask"].to(ed)], 1)
    with torch.no_grad(): gt = _generate_answer(trainer, gi, ivl)
    attn = trainer.compute_model_attention_single(user_id, image, profile_text, gt)
    if torch.is_tensor(attn): attn = attn.detach().float().cpu().numpy()
    s = float(attn.sum()); return (attn/s if s>1e-9 else np.ones(NUM_SLOTS, dtype=np.float32)/NUM_SLOTS), gt

def compute_baseline_rollout_freegeneration(trainer, image, profile_text):
    trainer.model.eval()
    mi, ids, _ = _build_baseline_inputs(trainer, image, profile_text, for_generate=True)
    with torch.no_grad(): gt = _generate_answer(trainer, mi, trainer.model_type == "internvl")
    return compute_baseline_rollout_attention(trainer, image, profile_text, gt), gt

def compute_option_logprobs(trainer, image, profile_text, user_id=None):
    trainer.model.eval(); ivl = trainer.model_type == "internvl"
    if user_id is not None:
        sp = trainer.prompt_basis([user_id])[0]
        inp, _ = trainer.build_inputs_with_soft_prompt(image=image, profile_text=profile_text, soft_prompt_embeds=sp, answer_text=None)
        fwd = {k: v for k, v in inp.items() if k != "labels" and not k.startswith("_")}
    else:
        fwd, _, _ = _build_baseline_inputs(trainer, image, profile_text, for_generate=True)
        fwd = {k: v for k, v in fwd.items() if not k.startswith("_")}
    with torch.no_grad(): out = (trainer.model.language_model if ivl else trainer.model)(**fwd)
    lp = torch.log_softmax(out.logits[0,-1].float(), dim=-1)
    return np.array([lp[tid].item() for tid in trainer._answer_token_ids])