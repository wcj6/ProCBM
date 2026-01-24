import torch
from torch import nn
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

from .utils import FFN


def count_keys_and_values(data_dict):
    """
    Count number of attributes and total number of concepts.
    """
    num_keys = len(data_dict)
    total_values = sum(len(values) for values in data_dict.values())
    return num_keys, total_values


# =========================
# PR-CBM Model
# =========================
class prcbm(nn.Module):
    """
    PR-CBM: Preference-Refined Concept Bottleneck Model

    Core idea:
    - Use CLIP-style vision-language model as backbone
    - Treat concepts as queries
    - Iteratively refine concept queries through multi-layer visual features
    - Predict class labels via concept logits
    """

    def __init__(self, concept_list, model_name='biomedclip', config=None):
        super().__init__()

        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config

        # -------------------------------------------------
        # Backbone & tokenizer
        # -------------------------------------------------
        if self.model_name == 'biomedclip':
            self.model, preprocess = create_model_from_pretrained(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self.tokenizer = get_tokenizer(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
        elif self.model_name == 'openclip':
            self.model, preprocess = create_model_from_pretrained(
                'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
            )
            self.tokenizer = get_tokenizer(
                'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
            )
        else:
            raise ValueError(f'Unknown model_name: {self.model_name}')

        # store preprocess in config for external usage
        config.preprocess = preprocess
        self.model.cuda()

        # -------------------------------------------------
        # Build concept text embeddings (frozen)
        # -------------------------------------------------
        concept_keys = list(concept_list.keys())
        _, num_concepts = count_keys_and_values(concept_list)

        self.concept_token_dict = {}
        for key in concept_keys:
            # dataset-specific prompt prefix
            if config.dataset == 'isic2018':
                prefix = f"this is a dermoscopic image, the {key} of the lesion is "
            elif config.dataset == 'cmmd':
                prefix = f"this is a x-ray image, the {key} of the breast cancer is "
            elif config.dataset == 'mimic_cxr':
                prefix = f"this is a x-ray image, the {key} of the lesion is "
            elif config.dataset == 'idrid':
                prefix = f"this is a Diabetic Retinopathy, the {key} of the lesion is "
            elif config.dataset == 'busi':
                prefix = f"this is a ultrasound image, the {key} of the lesion is "
            elif config.dataset == 'cm':
                prefix = f"this is a x-ray image, the {key} of the lesion is "
            elif config.dataset == 'nct':
                prefix = f"this is a histopathological image, the {key} of the lesion is "
            elif config.dataset == 'siim':
                prefix = f"this is a x-ray image, the {key} of the pneumothorax lesion is "
            else:
                raise ValueError(f'Unknown dataset: {config.dataset}')

            attr_concept_list = concept_list[key]
            text_inputs = [prefix + c for c in attr_concept_list]

            tokens = self.tokenizer(text_inputs).cuda()
            _, text_feats, logit_scale = self.model(None, tokens)

            # store frozen concept embeddings
            self.concept_token_dict[key] = text_feats.detach()

        self.logit_scale = logit_scale.detach()

        # -------------------------------------------------
        # Register visual feature hooks (layer-wise)
        # -------------------------------------------------
        self.visual_features = []
        self.hook_list = []

        def hook_fn(module, input, output):
            # save visual tokens of each transformer block
            self.visual_features.append(output)

        if self.model_name == 'biomedclip':
            for layer in self.model.visual.trunk.blocks:
                self.hook_list.append(layer.register_forward_hook(hook_fn))
        else:
            for layer in self.model.visual.transformer.resblocks:
                self.hook_list.append(layer.register_forward_hook(hook_fn))

        # -------------------------------------------------
        # Query-memory interaction modules
        # -------------------------------------------------
        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(7, 768)),
            requires_grad=False
        )

        self.linear_proj = nn.Linear(768, 512)
        self.proj = nn.Linear(768, 512, bias=False)

        self.query_cross_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=4, batch_first=True
        )

        self.global_gate_proj = nn.Linear(1536, 512)
        self.update_proj = nn.Linear(1024, 512)

        # -------------------------------------------------
        # Semantic aggregation & classifier
        # -------------------------------------------------
        self.semantic_proj = nn.Sequential(
            nn.Linear(512 * 12, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )

        self.pooler = nn.AdaptiveAvgPool1d(num_concepts)
        self.cls_head = nn.Linear(num_concepts, config.num_class)

        # -------------------------------------------------
        # Freeze strategy
        # -------------------------------------------------
        for p in self.model.text.parameters():
            p.requires_grad = False
        for p in self.model.visual.trunk.parameters():
            p.requires_grad = True

    # =========================
    # Optimizer helpers
    # =========================
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()

    def get_bridge_params(self):
        param_list = []
        modules = [
            self.linear_proj,
            self.proj,
            self.query_cross_attn,
            self.global_gate_proj,
            self.update_proj,
            self.semantic_proj,
            self.cls_head
        ]
        for m in modules:
            param_list.extend(m.parameters())
        return param_list

    # =========================
    # Forward
    # =========================
    def forward(self, imgs):
        """
        Forward pipeline:
        1. Extract multi-layer visual tokens via hooks
        2. Initialize concept queries
        3. Iteratively update queries with visual evidence
        4. Compute concept logits and class logits
        """
        self.visual_features.clear()
        _ = self.model(imgs, None)

        B = imgs.size(0)
        concept_keys = list(self.concept_token_dict.keys())

        # concatenate all concept embeddings
        concept_embeds = torch.cat(
            [self.concept_token_dict[k] for k in concept_keys], dim=0
        )
        query_memory = concept_embeds.unsqueeze(0).repeat(B, 1, 1)

        vision_memory_trace = []

        # -------------------------------------------------
        # Layer-wise query refinement
        # -------------------------------------------------
        for feats in self.visual_features:
            cls_token = feats[:, 0:1, :]
            patch_tokens = feats[:, 1:, :]

            pooled = self.pooler(
                patch_tokens.transpose(1, 2)
            ).transpose(1, 2)

            visual_tokens = torch.cat([cls_token, pooled], dim=1)
            visual_tokens = self.linear_proj(visual_tokens)

            # dot-product attention (concept queries over visual tokens)
            attn_scores = torch.bmm(
                query_memory, visual_tokens.transpose(1, 2)
            )
            attn_weights = F.softmax(attn_scores, dim=-1)
            fuse_out = torch.bmm(attn_weights, visual_tokens)

            vision_memory_trace.append(fuse_out)

            # gated memory update
            cls_context = self.proj(cls_token).expand_as(query_memory)
            gate = torch.sigmoid(
                self.global_gate_proj(
                    torch.cat([fuse_out, query_memory, cls_context], dim=-1)
                )
            )
            update = torch.tanh(
                self.update_proj(torch.cat([fuse_out, query_memory], dim=-1))
            )

            query_memory = gate * query_memory + (1 - gate) * update

        # -------------------------------------------------
        # Concept prediction
        # -------------------------------------------------
        vision_semantic = F.normalize(vision_memory_trace[-1], dim=-1)

        image_logits_dict = {}
        start_idx = 0
        for key in concept_keys:
            concept_anchor = self.concept_token_dict[key]
            end_idx = start_idx + concept_anchor.size(0)

            anchor = concept_anchor.repeat(B, 1, 1).permute(0, 2, 1)
            image_logits_dict[key] = (
                self.logit_scale
                * vision_semantic[:, start_idx:end_idx, :]
                @ anchor
            ).mean(1)

            start_idx = end_idx

        image_logits = torch.cat(
            [image_logits_dict[k] for k in concept_keys], dim=-1
        )
        cls_logits = self.cls_head(image_logits)

        # -------------------------------------------------
        # Consistency regularization
        # -------------------------------------------------
        loss_consistency = 0
        for i in range(1, len(vision_memory_trace)):
            loss_consistency += F.mse_loss(
                vision_memory_trace[i],
                vision_memory_trace[i - 1].detach()
            )

        return cls_logits, image_logits_dict, loss_consistency