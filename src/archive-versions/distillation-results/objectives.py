from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Recipe:
    temperature: float = 4
    logit: float = 0
    feature: str = ""
    scales: tuple = (4,)
    region: str = "uniform"
    teacher: str = "attention"
    student: str = "haar"


kl = Recipe(logit=0.5)
cwd = Recipe(feature="cwd")
dual = replace(kl, feature="cwd")
RECIPES = {
    "supervised": Recipe(),
    "pixel_kl": kl,
    "pixel_kl_temp2": replace(kl,temperature=2),
    "pixel_kl_temp6": replace(kl,temperature=6),
    "pixel_kl_weight025": replace(kl,logit=0.25),
    "pixel_kl_weight1": replace(kl,logit=1),
    "feature_mse_quarter": Recipe(feature="mse"),
    "feature_cwd_quarter": cwd,
    "feature_cwd_eighth": replace(cwd,scales=(8,)),
    "feature_cwd_multiscale": replace(cwd,scales=(4,8)),
    "pixel_kl_feature_mse": replace(kl,feature="mse"),
    "pixel_kl_feature_cwd": dual,
    "pixel_kl_feature_cwd_multiscale": replace(dual,scales=(4,8)),
    "edge_kl_feature_cwd": replace(dual,region="edge2"),
    "edge_kl_narrow": replace(kl,region="edge2"),
    "edge_kl_wide": replace(kl,region="edge5"),
    "balanced_kl": replace(kl,region="balanced"),
    "teacher_correct_kl": replace(kl,region="correct"),
    "tailor_features": Recipe(feature="pca"),
    "tailor_logits": replace(kl,region="tailor"),
    "tailor_joint": replace(kl,region="tailor",feature="pca"),
    "plain_teacher_kl": replace(kl,teacher="plain"),
    "gray_supervised": Recipe(student="gray"),
    "gray_dual": replace(dual,student="gray"),
}
REPEAT = ("supervised","pixel_kl","pixel_kl_feature_cwd","tailor_joint")


def dilate(mask, radius):
    return F.max_pool2d(mask[:,None].float(),2*radius+1,stride=1,padding=radius)[:,0].bool()


def boundary(mask):
    return mask & dilate(~mask,1)


class Objective(nn.Module):
    def __init__(self, recipe, teacher_channels):
        super().__init__()
        self.recipe = recipe
        self.project = nn.ModuleDict({
            str(scale): nn.Conv2d(64,16 if recipe.feature=="pca" else teacher_channels[scale],1)
            for scale in recipe.scales if recipe.feature
        })

    def forward(self, logits, student_features, target, teacher_logits=None,
                teacher_features=None, progress=0):
        logits = logits.float()
        p = logits.softmax(1)[:,1]
        fg = target.float()
        intersection = (p*fg).sum()
        supervised = F.cross_entropy(logits,target) + 1-(2*intersection+1)/(p.sum()+fg.sum()+1)
        logit_loss = feature_loss = logits.new_zeros(())
        recipe = self.recipe
        if recipe.logit:
            t = recipe.temperature
            q = (teacher_logits.float()/t).softmax(1)
            weights = torch.ones_like(fg)
            if recipe.region.startswith("edge"):
                radius = int(recipe.region.removeprefix("edge"))
                weights = weights + 3*dilate(boundary(target.bool()),radius)
            elif recipe.region=="balanced":
                prevalence = fg.mean().clamp(0.02,0.98)
                weights = torch.where(target.bool(),0.5/prevalence,0.5/(1-prevalence))
            elif recipe.region=="correct":
                weights = torch.where(teacher_logits.argmax(1)==target,1.0,0.1)
            elif recipe.region=="tailor":
                edge = dilate(boundary(target.bool()),2)
                hard = F.one_hot(target,2).permute(0,3,1,2).float()
                blend = edge[:,None]*(0.35-0.15*progress)
                q = q*(1-blend) + hard*blend
            kl_map = F.kl_div(F.log_softmax(logits/t,dim=1),q,reduction="none").sum(1)
            logit_loss = (kl_map*weights).sum()/weights.sum()*t*t
        if recipe.feature:
            terms = []
            for scale in recipe.scales:
                a = self.project[str(scale)](student_features[scale]).float()
                b = teacher_features[scale].float()
                if recipe.feature=="pca":
                    # Teacher correlation basis from the current training minibatch.
                    # Canonicalize the eigenvector signs and favor higher-variance directions.
                    with torch.no_grad(), torch.autocast(device_type=b.device.type,enabled=False):
                        samples = b.permute(0,2,3,1).reshape(-1,b.shape[1])
                        standardized = (samples-samples.mean(0))/samples.std(0,unbiased=False).clamp_min(1e-4)
                        _, vectors = torch.linalg.eigh(standardized.T @ standardized / len(samples))
                        basis = vectors[:,-16:].flip(1)
                        sign = basis.gather(0,basis.abs().argmax(0,keepdim=True)).sign()
                        b = (standardized @ (basis*sign)).reshape(b.shape[0],*b.shape[-2:],16)
                        b = b.permute(0,3,1,2).contiguous()
                if recipe.feature=="mse":
                    terms.append(F.mse_loss(a,b))
                else:
                    log_a = (a.flatten(2)/2).log_softmax(2)
                    prob_b = (b.flatten(2)/2).softmax(2)
                    terms.append(F.kl_div(log_a,prob_b,reduction="none").sum(2).mean()*4)
            feature_loss = torch.stack(terms).mean()
        return supervised + recipe.logit*logit_loss + 0.1*feature_loss, (supervised,logit_loss,feature_loss)
