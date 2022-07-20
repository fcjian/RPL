import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, category_descriptions, clip_model):
        super().__init__()
        classnames = [category_description.split("_which_is_")[0] for category_description in category_descriptions]
        classdefs = [category_description.split("_which_is_")[1] for category_description in category_descriptions]

        n_cls = len(classnames)
        n_ctx1 = cfg.TRAINER.RPL.N_CTX
        n_ctx2 = cfg.TRAINER.RPL.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        print("Initializing a generic context")
        ctx_vectors1 = torch.empty(n_ctx1, ctx_dim, dtype=dtype)
        ctx_vectors2 = torch.empty(n_ctx2, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors1, std=0.02)
        nn.init.normal_(ctx_vectors2, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx1)
        prompt_suffix = " ".join(["X"] * n_ctx2)

        print(f'Initial context1: "{prompt_prefix}"')
        print(f'Initial context2: "{prompt_suffix}"')
        print(f"Number of context words (tokens1): {n_ctx1}")
        print(f"Number of context words (tokens2): {n_ctx2}")

        self.ctx1 = nn.Parameter(ctx_vectors1)  # to be optimized
        self.ctx2 = nn.Parameter(ctx_vectors2)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        classdefs = [name.replace("_", " ") for name in classdefs]
        def_lens = [len(_tokenizer.encode(name)) for name in classdefs]

        prompts = [prompt_prefix + " " + name + " " + prompt_suffix + " " + classdef + "." for name, classdef in
                   zip(classnames, classdefs)]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("embedding", embedding)  # SOS

        self.n_cls = n_cls
        self.n_ctx1 = n_ctx1
        self.n_ctx2 = n_ctx2
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.def_lens = def_lens

    def forward(self):
        ctx1 = self.ctx1
        ctx2 = self.ctx2
        if ctx1.dim() == 2:
            ctx1 = ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx2 = ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)

        embedding = self.embedding

        n_ctx1 = self.n_ctx1
        n_ctx2 = self.n_ctx2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            def_len = self.def_lens[i]
            prefix_i = embedding[i: i + 1, :1, :]
            ctx1_i = ctx1[i:i + 1]
            name_index = 1 + n_ctx1
            name_i = embedding[i: i + 1, name_index:name_index + name_len, :]
            ctx2_i = ctx2[i:i + 1]
            def_index = 1 + n_ctx1 + name_len + n_ctx2
            def_i = embedding[i: i + 1, def_index:def_index + def_len, :]
            suf_index = 1 + n_ctx1 + name_len + n_ctx2 + def_len
            suffix_i = embedding[i: i + 1, suf_index:, :]
            prompt = torch.cat(
                [
                    prefix_i,
                    ctx1_i,
                    name_i,
                    ctx2_i,
                    def_i,
                    suffix_i,
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class RPL(TrainerX):
    """Context Optimization (RPL).

    PromptDet
    https://arxiv.org/abs/2203.16513
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.RPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.RPL.PREC == "fp32" or cfg.TRAINER.RPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.RPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.RPL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
