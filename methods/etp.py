import sys, warnings

warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse
import lightning.pytorch as pl
from torch import optim
from torch.distributions import Dirichlet, Normal
from torch.distributions.kl import kl_divergence
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor, nn
from copy import deepcopy
import torch
import math
import numpy as np
from src.training import ClassificationRoutine
from src.metrics.ood import TUOODCriterion, OODCriterionInputType
from lightning.pytorch.utilities.types import STEP_OUTPUT
from src.utils import (
    add_common_args, print_gpu_config, load_gpu_config, get_datamodule, get_model,
)
from src.training.experiment import (
    add_experiment_args,
    evaluate,
    make_trainer,
    run_repeated,
)

METHOD_NAME = "etp"

class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_prec=10, map=True):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.prior_prec = prior_prec
        self.map = map
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.bias.data.zero_()

    def KL(self, loguniform=False):
        if loguniform:
            k1 = 0.63576
            k2 = 1.87320
            k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * torch.log(self.mu_w.abs() + 1e-8)
            kl = -torch.sum(
                k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * F.softplus(-log_alpha) - k1)
        else:
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            kl = (
                0.5
                * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                    - logsig2_w - 1 - np.log(self.prior_prec)).sum())
        return kl

    def forward(self, input):
        # Sampling free forward pass only if MAP prediction and no training rounds
        if self.map and not self.training:
            return F.linear(input, self.mu_w, self.bias)
        else:
            mu_out = F.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_in)
            + " -> "
            + str(self.n_out)
            + ")"
        )

class ETPHyperParams:
    def __init__(self, n_classes=None):
        super(ETPHyperParams, self).__init__()
        self.memory_learning_rate = 0.9  # seu 0.9
        self.memory_size = 20  #  seu 20
        self.anneal_factor = 1e-5
        self.memo_variance = 0.01 #  seu 0.01

class EvidentialTuringProcess(nn.Module):
    def __init__(self, arch=None, n_classes=None):
        super(EvidentialTuringProcess, self).__init__()

        self.arch = deepcopy(arch)
        self.n_classes = n_classes
        self.hyperparams = ETPHyperParams(self.n_classes)
        self.memory = nn.Parameter(torch.Tensor(self.hyperparams.memory_size, self.n_classes),
                                   requires_grad=False)
        self.memory.data.normal_(0, 0.01)
        self.memory.data.pow_(2)

        # tmp = torch.empty(self.n_classes, self.hyperparams.memory_size)
        # nn.init.orthogonal_(tmp)
        # self.memory.data.copy_(tmp.t())

        # self.fc1_enc_to_pred = nn.Linear(self.n_classes * 2, self.n_classes)
        # self.fc1_key = nn.Linear(self.n_classes, self.n_classes)

        self.fc1_enc_to_pred = VBLinear(self.n_classes * 2, self.n_classes, prior_prec=10)
        self.fc1_key = VBLinear(self.n_classes, self.n_classes, prior_prec=10)


    def update_memory(self, x_embed, y, max_size=128):
        n_context = max_size
        x_given_embed = x_embed[:n_context, :]
        y_given = y[:n_context].view(-1, 1)
        mem_sample = self.get_memory_sample()
        new_element = F.one_hot(y_given, self.n_classes).view(-1, self.n_classes) + torch.softmax(x_given_embed,
                                                                                                         1)
        weight_new_element = self.get_attention_weights(x_given_embed, mem_sample)
        add_new_element = torch.mm(weight_new_element.transpose(0, 1), new_element)
        gamma = self.hyperparams.memory_learning_rate
        mem_offset = self.memory * (gamma - 1) + add_new_element * (1 - gamma)
        self.memory.data.add_(mem_offset)
        self.memory.data.tanh_()  
    


    def get_memory_sample(self):
        if not self.training:
            return self.memory
        sig2 = self.hyperparams.memo_variance
        sig2_vec = torch.ones(self.memory.shape, device=self.memory.device) * sig2
        return Normal(self.memory, sig2_vec).rsample()


    def get_attention_weights(self, x_embed, mem_sample):
        keys = self.fc1_key(mem_sample)
        kq = torch.mm(x_embed, keys.transpose(0, 1)) / np.sqrt(self.n_classes)
        return F.softmax(kq, 1)

    def get_attention(self, x_embed):
        mem_sample = self.get_memory_sample()
        weights = self.get_attention_weights(x_embed, mem_sample)
        return torch.mm(weights, mem_sample)
    
    def forward(self, input):
        x_embed = self.arch(input)
        attention = self.get_attention(x_embed)
        logit = self.fc1_enc_to_pred(torch.cat((x_embed, attention), dim=1))
        return logit.clamp(max=15)

    def kl_dirichlet(self, alpha, beta):
        q = Dirichlet(alpha)
        p = Dirichlet(beta)
        return kl_divergence(q, p)

    def KL(self):
        return sum(module.KL() for module in self.modules() if isinstance(module, VBLinear))
    
    def loss(self, x, y):
        y_one_hot = F.one_hot(y, self.n_classes).view(-1, self.n_classes)
        x_embed_pre = self.arch(x)
        attention = self.get_attention(x_embed_pre)
        logit = self.fc1_enc_to_pred(torch.cat((x_embed_pre, attention), dim=1))
        x_embed = logit.clamp(max=15)
        self.update_memory(x_embed_pre, y)
        alpha = torch.exp(x_embed)
        S = alpha.sum(1, keepdims=True)
        fit_term = (y_one_hot * (torch.digamma(S + 1e-8) - torch.digamma(alpha + 1e-8))).sum(axis=1)
        reg_term = self.kl_dirichlet(alpha, torch.exp(attention))
        loss = (fit_term + reg_term * self.hyperparams.anneal_factor).mean()
        return loss + self.KL() / self.dataset_size
    
class ETPOODCriterion(TUOODCriterion):
    input_type    = OODCriterionInputType.LOGIT
    ensemble_only = False
    single_only   = True

    def forward(self, logits: Tensor) -> Tensor:
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        alpha = torch.exp(logits.clamp(max=15))
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  

class _Loss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("_Loss should never be called.")

class ETPClassificationRoutine(ClassificationRoutine):
    def __init__(self, model: EvidentialTuringProcess, acc_gate: float = 0.98, **kwargs):
        kwargs.setdefault("ood_criterion", ETPOODCriterion())
        super().__init__(model=model, loss=_Loss(), **kwargs)
        self._etp = model
        self.acc_gate = acc_gate

    def training_step(self, batch: tuple[Tensor, Tensor], *args, **kwargs) -> STEP_OUTPUT:
        batch = self._apply_mixup(batch)
        inputs, targets = self.format_batch_fn(batch)
        loss = self._etp.loss(inputs, targets)
        # clip_grad_norm_(self._etp.parameters(), max_norm=5.0)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        res_dict = self.val_cls_metrics.compute()   # 先取值，父类会 reset
        acc = res_dict["val/cls/Acc"]
        ece = res_dict["val/cal/ECE"]
        super().on_validation_epoch_end()           # 正常 log + reset
        # Acc 达标后按 ECE 选，否则给一个极差的分
        score = ece if acc >= self.acc_gate else torch.tensor(1e6, device=ece.device)
        self.log("val/selection_score", score, logger=True, sync_dist=True)

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)


class AnnealCallback(pl.Callback):
    def __init__(self, model: EvidentialTuringProcess, start: float, end: float, max_epochs: int):
        self.model      = model
        self.start      = start
        self.end        = end
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        progress = min(trainer.current_epoch / max(self.max_epochs - 1, 1), 1.0)
        new_factor = self.start + (self.end - self.start) * progress
        self.model.hyperparams.anneal_factor = new_factor

def run_once(args, seed, run_dir):
    datamodule = get_datamodule(
        args, val_split=args.val_split, eval_ood=True,
    )
    backbone = get_model(args.backbone, datamodule.num_channels, datamodule.num_classes)
    model = EvidentialTuringProcess(backbone, n_classes=datamodule.num_classes)
    datamodule.setup("fit")
    model.dataset_size = len(datamodule.train_dataloader().dataset)
    routine = ETPClassificationRoutine(
        model=model,
        num_classes=datamodule.num_classes,
        optim_recipe=optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005),
        eval_ood=True,
        acc_gate=args.acc_gate,
        save_in_csv=True,
    )
    trainer = make_trainer(
        args,
        run_dir,
        monitor="val/selection_score",
        mode="min",
        callbacks=[
            AnnealCallback(
                model,
                start=args.anneal_start,
                end=args.anneal_end,
                max_epochs=args.epochs,
            )
        ],
        gradient_clip_val=args.grad_clip,
    )
    trainer.fit(model=routine, datamodule=datamodule)
    return evaluate(args, trainer, routine, datamodule, ckpt_path="best")


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--acc-gate", type=float, default=0.98)
    parser.add_argument("--anneal-start", type=float, default=1e-5)
    parser.add_argument("--anneal-end", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
