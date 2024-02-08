import pytorch_lightning as pl
import torch.optim as optim
from functions import *
import torch.nn as nn
import torch


EMBED_DIM = 512


class Projection(nn.Module):

    def __init__(self, d_in, d_out, p=0.5):

        super().__init__()

        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):

        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisionEncoder(nn.Module):

    def __init__(self, base, d_out):

        super().__init__()

        self.base = base
        d_in = self.base.config.hidden_size
        self.projection = Projection(d_in, d_out)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        inputs = {'pixel_values': x}
        out = self.base.forward(**inputs)
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(nn.Module):

    def __init__(self, base, d_out):

        super().__init__()

        self.base = base
        d_in = self.base.config.hidden_size
        self.projection = Projection(d_in, d_out)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base(x)
        # out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class CustomCLIP(pl.LightningModule):

    def __init__(self,
                 vision_model,
                 text_model,
                 similarity_fn=similarity_function,
                 loss_fn=nn.CrossEntropyLoss()
                 ):

        super(CustomCLIP, self).__init__()

        self.vision_encoder = VisionEncoder(vision_model, EMBED_DIM)
        self.text_encoder = TextEncoder(text_model, EMBED_DIM)
        self.similarity_function = similarity_fn
        self.loss_fn = loss_fn

    def forward(self, x):

        image = x['pixel_values']
        caption = x['labels']

        image_embedding = self.vision_encoder(image)
        text_embedding = self.text_encoder(caption)
        similarity_score = self.similarity_fn(image_embedding, text_embedding)

        return similarity_score

    def training_step(self, batch, batch_idx):

        similarity_score = self(batch)
        loss = self.loss_fn(similarity_score)

        self.log("training_loss", loss, on_step=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
