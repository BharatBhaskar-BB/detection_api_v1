# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Finalised single-file predictor for SAM3 semantic (text-prompted) segmentation.
Only SAM3SemanticPredictor and the code it directly calls are kept.
"""

from __future__ import annotations

# ── standard / third-party ────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── utils ─────────────────────────────────────────────────────────────────────
from dataio.augment import LetterBox
from pipeline.predictor import BasePredictor
from pipeline.results import Results
from helpers import DEFAULT_CFG, ops
from helpers.patches import torch_load
from helpers.torch_utils import select_device, smart_inference_mode

# ── SAM3 model components ─────────────────────────────────────────────────────
from core.sam_blocks import PositionEmbeddingSine
from network.transformer import MLP
from core.decoder import TransformerDecoder, TransformerDecoderLayer
from core.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from core.geometry_encoders import SequenceGeometryEncoder
from core.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from core.model_misc import DotProductScoring, TransformerWrapper
from core.necks import Sam3DualViTDetNeck
from core.sam3_image import SAM3SemanticModel
from core.text_encoder_ve import VETextEncoder
from core.vitdet import ViT
from core.vl_combiner import SAM3VLBackbone


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Prompt helpers  (minimal extract from geometry_encoders.py)
# ══════════════════════════════════════════════════════════════════════════════

def _is_right_padded(mask: torch.Tensor) -> bool:
    """True when 1-padded values are all on the right side of the mask."""
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def _concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """Merge two right-padded sequences into one contiguous right-padded sequence.

    Tensors are sequence-first, batch-second; masks are batch-first (1 = padded).

    Args:
        seq1:  (S1, B, C)
        mask1: (B, S1)
        seq2:  (S2, B, C)
        mask2: (B, S2)
        return_index: if True, also return scatter positions of seq2 in the output.
    """
    S1, B, C = seq1.shape
    S2 = seq2.shape[0]

    torch._assert(_is_right_padded(mask1), "mask1 is not right-padded")
    torch._assert(_is_right_padded(mask2), "mask2 is not right-padded")

    len1      = (~mask1).sum(dim=-1)
    final_len = len1 + (~mask2).sum(dim=-1)
    max_len   = S1 + S2

    out_mask = torch.arange(max_len, device=seq2.device)[None].repeat(B, 1) >= final_len[:, None]
    out_seq  = torch.zeros((max_len, B, C), device=seq2.device, dtype=seq2.dtype)
    out_seq[:S1] = seq1

    idx     = torch.arange(S2, device=seq2.device)[:, None].repeat(1, B) + len1[None]
    out_seq = out_seq.scatter(0, idx[:, :, None].expand(-1, -1, C), seq2)

    return (out_seq, out_mask, idx) if return_index else (out_seq, out_mask)


class Prompt:
    """Geometric box-prompt container (pytorch sequence-first convention).

    Shapes:
        box_embeddings : (N, B, 4)  – normalised CxCyWH coordinates
        box_mask       : (B, N)     – True = padded / ignored slot
        box_labels     : (N, B)     – 1 = positive, 0 = negative
    """

    def __init__(self, box_embeddings=None, box_mask=None, box_labels=None):
        if box_embeddings is None:
            self.box_embeddings = self.box_mask = self.box_labels = None
            return

        N, B, dev = box_embeddings.shape[0], box_embeddings.shape[1], box_embeddings.device

        if box_labels is None:
            box_labels = torch.ones(N, B, device=dev, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(B, N, device=dev, dtype=torch.bool)

        assert box_embeddings.shape[-1] == 4,             "box_embeddings last dim must be 4"
        assert list(box_embeddings.shape[:2]) == [N, B],  f"box_embeddings shape mismatch: {box_embeddings.shape}"
        assert list(box_mask.shape)   == [B, N],          f"box_mask shape mismatch: {box_mask.shape}"
        assert list(box_labels.shape) == [N, B],          f"box_labels shape mismatch: {box_labels.shape}"

        self.box_embeddings = box_embeddings
        self.box_mask       = box_mask
        self.box_labels     = box_labels

    def append_boxes(self, boxes, labels=None, mask=None):
        """Append new box prompts.

        Args:
            boxes:  (N_new, B, 4) – normalised coordinates
            labels: (N_new, B)    – pos/neg labels   (default: all positive)
            mask:   (B, N_new)    – attention mask    (default: all active)
        """
        N, B, dev = boxes.shape[0], boxes.shape[1], boxes.device

        if labels is None:
            labels = torch.ones(N, B, device=dev, dtype=torch.long)
        if mask is None:
            mask = torch.zeros(B, N, device=dev, dtype=torch.bool)

        if self.box_embeddings is None:
            self.box_embeddings, self.box_labels, self.box_mask = boxes, labels, mask
            return

        assert boxes.shape[1]        == self.box_embeddings.shape[1], "Batch-size mismatch"
        assert list(boxes.shape[:2]) == list(labels.shape[:2]),        "boxes / labels shape mismatch"

        self.box_labels, _ = _concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask,
            labels.unsqueeze(-1),          mask,
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = _concat_padded_sequences(
            self.box_embeddings, self.box_mask, boxes, mask,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SAM3 image-model builder  (minimal extract from build_sam3.py)
# ══════════════════════════════════════════════════════════════════════════════

def _create_vision_backbone(compile_mode=None) -> Sam3DualViTDetNeck:
    """Build SAM3 ViT-L backbone + FPN neck."""
    pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True, temperature=10000)
    vit = ViT(
        img_size=1008, pretrain_img_size=336, patch_size=14,
        embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625,
        norm_layer="LayerNorm", drop_path_rate=0.1, qkv_bias=True,
        use_abs_pos=True, tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(),
        use_rope=True, use_interp_rope=True, window_size=24,
        pretrain_use_cls_token=True, retain_cls_token=False,
        ln_pre=True, ln_post=False, return_interm_layers=False,
        bias_patch_embed=False, compile_mode=compile_mode,
    )
    return Sam3DualViTDetNeck(
        position_encoding=pos_enc, d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit, add_sam2_neck=True,
    )


def _create_sam3_transformer() -> TransformerWrapper:
    """Build SAM3 encoder-decoder transformer."""
    def _mha(**kw):
        return nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, **kw)

    encoder = TransformerEncoderFusion(
        layer=TransformerEncoderLayer(
            d_model=256, dim_feedforward=2048, dropout=0.1, pre_norm=True,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False, pos_enc_at_cross_attn_queries=False,
            self_attention=_mha(batch_first=True), cross_attention=_mha(batch_first=True),
        ),
        num_layers=6, d_model=256, num_feature_levels=1, frozen=False,
        use_act_checkpoint=True, add_pooled_text_to_img_feat=False, pool_text_with_mask=True,
    )
    decoder = TransformerDecoder(
        layer=TransformerDecoderLayer(
            d_model=256, dim_feedforward=2048, dropout=0.1,
            cross_attention=_mha(), n_heads=8, use_text_cross_attention=True,
        ),
        num_layers=6, num_queries=200, return_intermediate=True, box_refine=True,
        num_o2m_queries=0, dac=True, boxRPB="log", d_model=256, frozen=False,
        interaction_layer=None, dac_use_selfatt_ln=True,
        use_act_checkpoint=True, presence_token=True,
    )
    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint_path: str):
    """Load SAM3 image-model weights; silently ignores missing / unexpected keys."""
    with open(checkpoint_path, "rb") as f:
        ckpt = torch_load(f)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    state = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    model.load_state_dict(state, strict=False)
    return model


def build_sam3_image_model(checkpoint_path: str, compile: bool = False) -> SAM3SemanticModel:
    """Construct, load and return an eval-mode SAM3 semantic image model.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        compile:         Enable torch.compile on the vision backbone.
    """
    import clip

    compile_mode = "default" if compile else None
    pos_enc      = PositionEmbeddingSine(num_pos_feats=256, normalize=True, temperature=10000)

    model = SAM3SemanticModel(
        backbone=SAM3VLBackbone(
            visual=_create_vision_backbone(compile_mode),
            text=VETextEncoder(
                tokenizer=clip.tokenize,
                d_model=256, width=1024, heads=16, layers=24,
            ),
            scalp=1,
        ),
        transformer=_create_sam3_transformer(),
        input_geometry_encoder=SequenceGeometryEncoder(
            pos_enc=pos_enc,
            encode_boxes_as_points=False,
            boxes_direct_project=True, boxes_pool=True, boxes_pos_enc=True,
            d_model=256, num_layers=3,
            layer=TransformerEncoderLayer(
                d_model=256, dim_feedforward=2048, dropout=0.1,
                pos_enc_at_attn=False, pre_norm=True,
                pos_enc_at_cross_attn_queries=False, pos_enc_at_cross_attn_keys=True,
            ),
            use_act_ckpt=True, add_cls=True, add_post_encode_proj=True,
        ),
        segmentation_head=UniversalSegmentationHead(
            hidden_dim=256, upsampling_stages=3,
            aux_masks=False, presence_head=False,
            dot_product_scorer=None, act_ckpt=True,
            cross_attend_prompt=nn.MultiheadAttention(num_heads=8, dropout=0, embed_dim=256),
            pixel_decoder=PixelDecoder(
                num_upsampling_stages=3, interpolation_mode="nearest",
                hidden_dim=256, compile_mode=compile_mode,
            ),
        ),
        dot_prod_scoring=DotProductScoring(
            d_model=256, d_proj=256,
            prompt_mlp=MLP(
                input_dim=256, hidden_dim=2048, output_dim=256,
                num_layers=2, residual=True, out_norm=nn.LayerNorm(256),
            ),
        ),
        num_feature_levels=1, o2m_mask_predict=True,
        use_instance_query=False, multimask_output=True,
    )

    model = _load_checkpoint(model, checkpoint_path)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Predictor hierarchy
# ══════════════════════════════════════════════════════════════════════════════

class Predictor(BasePredictor):
    """SAM base predictor: shared init, preprocessing, and model-setup logic."""

    stride = 32

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", batch=1))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = self.features = None
        self.prompts     = {}
        self.segment_all = False

    # --- preprocessing ---------------------------------------------------
    def preprocess(self, im):
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = torch.from_numpy(np.ascontiguousarray(im))
        im = im.to(self.device)
        if not_tensor:
            im = (im - self.mean) / self.std
        return im.half() if self.model.fp16 else im.float()

    def pre_transform(self, im):
        assert len(im) == 1, "SAM does not support batched inference"
        return [LetterBox(self.imgsz, auto=False, center=False)(image=x) for x in im]

    # --- model setup -----------------------------------------------------
    def setup_model(self, model=None, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        model.eval()
        self.model      = (model.half() if self.args.half else model.float()).to(device)
        self.device     = device
        self.mean       = torch.tensor([123.675, 116.28,  103.53 ]).view(-1, 1, 1).to(device)
        self.std        = torch.tensor([ 58.395,  57.12,   57.375]).view(-1, 1, 1).to(device)
        self.model.pt   = self.model.triton = False
        self.model.stride   = self.stride
        self.model.fp16     = self.args.half
        self.done_warmup    = True
        self.torch_dtype    = torch.float16 if self.model.fp16 else torch.float32

    # --- single-image API ------------------------------------------------
    def set_image(self, image):
        if self.model is None:
            self.setup_model()
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` supports one image only"
        for batch in self.dataset:
            self.features = self.get_im_features(self.preprocess(batch[1]))
            break

    def setup_source(self, source):
        if source is None:
            return
        super().setup_source(source, self.stride)
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM requires a square image size, got {self.imgsz}."
        )
        self.model.set_imgsz(self.imgsz)

    def reset_image(self):
        self.im = self.features = None


# ─────────────────────────────────────────────────────────────────────────────
class SAM3SemanticPredictor(Predictor):
    """SAM3 semantic predictor — text-prompted open-vocabulary segmentation.

    Overrides the full parent stack:
      • get_model        → build_sam3_image_model  (image model, not video)
      • setup_model      → SAM3 pixel normalisation [-1, 1]
      • get_im_features  → backbone-only forward (no memory)
      • pre_transform    → scale_fill LetterBox (no padding)
    """

    stride = 14

    # --- model -----------------------------------------------------------
    def get_model(self):
        return build_sam3_image_model(self.args.model, compile=self.args.compile)

    def setup_model(self, model=None, verbose=True):
        super().setup_model(model, verbose)
        # SAM3 uses symmetric [-1, 1] normalisation instead of ImageNet stats
        self.mean = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)
        self.std  = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)

    # --- preprocessing ---------------------------------------------------
    @smart_inference_mode()
    def get_im_features(self, im):
        """Run backbone only — no memory encoder needed for image-level inference."""
        return self.model.backbone.forward_image(im)

    def pre_transform(self, im):
        assert len(im) == 1, "SAM does not support batched inference"
        # scale_fill=True: stretch to square, no black-bar padding
        return [LetterBox(self.imgsz, auto=False, center=False, scale_fill=True)(image=x) for x in im]

    # --- prompt helpers --------------------------------------------------
    def _prepare_geometric_prompts(self, src_shape, bboxes=None, labels=None):
        """Convert xyxy pixel boxes → normalised CxCyWH tensors shaped (N, 1, 4)."""
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = ops.xyxy2xywh(bboxes[None] if bboxes.ndim == 1 else bboxes)
            bboxes[:, 0::2] /= src_shape[1]   # normalise x by width
            bboxes[:, 1::2] /= src_shape[0]   # normalise y by height
            if labels is None:
                labels = np.ones(bboxes.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert bboxes.shape[-2] == labels.shape[-1]
            bboxes, labels = bboxes.view(-1, 1, 4), labels.view(-1, 1)
        return bboxes, labels

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        """Return an empty Prompt (zero boxes) — used for pure text queries."""
        return Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=self.device),
            box_mask=torch.zeros(num_prompts, 0, device=self.device, dtype=torch.bool),
        )

    # --- core inference --------------------------------------------------
    def _inference_features(self, features, bboxes=None, labels=None,
                             text: list[str] | None = None):
        """Run the grounding forward pass given pre-extracted backbone features."""
        nc  = 1 if bboxes is not None else (len(text) if text is not None else len(self.model.names))
        geo = self._get_dummy_prompt(nc)

        if bboxes is not None:
            for i in range(len(bboxes)):
                geo.append_boxes(bboxes[[i]], labels[[i]])
            if text is None:
                text = ["visual"]

        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)

        return self.model.forward_grounding(
            backbone_out=features,
            text_ids=torch.arange(nc, device=self.device, dtype=torch.long),
            geometric_prompt=geo,
        )

    def inference(self, im, bboxes=None, labels=None,
                  text: list[str] | None = None, *args, **kwargs):
        """Entry point for the predict loop."""
        bboxes   = self.prompts.pop("bboxes", bboxes)
        labels   = self.prompts.pop("labels", labels)
        text     = self.prompts.pop("text",   text)
        features = self.get_im_features(im) if self.features is None else self.features
        bboxes, labels = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        return self._inference_features(features, bboxes, labels, text)

    # --- postprocess -----------------------------------------------------
    def postprocess(self, preds, img, orig_imgs):
        """Convert raw model outputs to Results objects.

        Steps:
          1. score = sigmoid(logit) × sigmoid(presence_logit)
          2. confidence threshold filter
          3. class-offset NMS
          4. mask bilinear upsample + binarise
          5. box denormalise to pixel coords
        """
        import torchvision

        pred_boxes  = preds["pred_boxes"]                          # (nc, Q, 4) normalised xywh
        pred_scores = (preds["pred_logits"].sigmoid()
                       * preds["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        pred_masks  = preds["pred_masks"]                          # (nc, Q, H, W)

        # Build (nc*Q, 6) boxes tensor: [x,y,x,y, score, cls]
        pred_cls   = (torch.arange(pred_scores.shape[0], dtype=pred_scores.dtype, device=pred_scores.device)
                      [:, None].expand_as(pred_scores))
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        # Confidence filter (applied jointly to masks and boxes)
        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = pred_masks[keep], pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        # Class-aware NMS (cast to float32 — nms_kernel doesn't support Half)
        nms_boxes = pred_boxes[:, :4] + pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)
        keep      = torchvision.ops.nms(nms_boxes.float(), pred_boxes[:, 4].float(), self.args.iou)
        pred_boxes, pred_masks = pred_boxes[keep], pred_masks[keep]

        names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for masks, boxes, orig_img, img_path in zip(
            [pred_masks], [pred_boxes], orig_imgs, self.batch[0]
        ):
            if masks.shape[0] == 0:
                masks, boxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = F.interpolate(masks.float()[None], orig_img.shape[:2], mode="bilinear")[0] > 0.5
                boxes[..., [0, 2]] *= orig_img.shape[1]   # x → pixel
                boxes[..., [1, 3]] *= orig_img.shape[0]   # y → pixel
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    # --- reset -----------------------------------------------------------
    def reset_prompts(self):
        """Clear pending prompts and cached text embeddings."""
        self.prompts = {}
        self.model.text_embeddings = {}