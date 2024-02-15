from transformers import ViTImageProcessor, DistilBertModel, DistilBertTokenizer, ViTModel
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import functions
import config
import torch


class Projection(pl.LightningModule):

    def __init__(self, d_in: int, d_out: int, p: float = 0.5):

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


class VisionEncoder(pl.LightningModule):

    def __init__(self):

        super().__init__()

        # Create the vision transformer
        self.preprocessor = ViTImageProcessor(config.VISION_MODEL)
        self.vision_model = ViTModel.from_pretrained(config.VISION_MODEL)

        # Take in and out dimensions of the last linear layer
        last_linear_layer = self.vision_model.encoder.layer[-1].output.dense
        last_linear_layer_out = last_linear_layer.out_features

        # Create a projection layer using the size of the vision models output features as in dimension
        self.projection = Projection(last_linear_layer_out, config.EMBED_DIM)

        # Freeze the vision model parameters
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def forward(self, x):

        # Pass the image to the preprocessor and get a tensor as result
        model_input = self.preprocessor(x, return_tensors="pt")

        # Give the tensor to the model and take the last hidden state
        model_output = self.vision_model(**model_input)
        last_hidden_states = model_output.last_hidden_state

        # Extract CLS token representation
        features = last_hidden_states[:, 0, :]

        # Project the last hidden state to the new embedding space
        projected_vec = self.projection(features)

        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class TextEncoder(pl.LightningModule):

    def __init__(self):

        super().__init__()

        # Create the text transformer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.TEXT_MODEL)
        self.text_model = DistilBertModel.from_pretrained(config.TEXT_MODEL)

        # Take in and out dimensions of the last linear layer
        last_linear_layer = self.text_model.transformer.layer[-1].output_layer_norm
        last_linear_layer_out = last_linear_layer.normalized_shape[0]

        # Create a projection layer using the size of the vision models output features as in dimension
        self.projection = Projection(last_linear_layer_out, config.EMBED_DIM)

        # Freeze the text model parameters
        for p in self.text_model.parameters():
            p.requires_grad = False

    def forward(self, x):

        # Pass the text to the tokenizer and get a dictionary containing a tensor as result
        model_input = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)

        # Give the tensor to the model and take  the last hidden state
        model_output = self.text_model(**model_input)
        last_hidden_states = model_output.last_hidden_state

        # Extract CLS token representation
        features = last_hidden_states[:, 0, :]

        # Project the last hidden state to the new embedding space
        projected_vec = self.projection(features)

        projection_len = torch.norm(projected_vec, dim=-1, keepdim=True)
        return projected_vec / projection_len


class CLIPLike(pl.LightningModule):

    def __init__(self):

        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.caption_encoder = TextEncoder()
        self.lr = config.LEARNING_RATE

    def common_step(self, batch):

        # Take images and text from the batch
        images, texts = batch

        # Create the embeddings
        image_embeds = self.vision_encoder(images)
        text_embeds = self.caption_encoder(texts)

        # Compute similarity and loss
        sim = functions.similarity_matrix(image_embeds, text_embeds)
        loss = functions.clip_loss(sim)
        img_acc, cap_acc = functions.metrics(sim)

        return loss, img_acc, cap_acc

    def training_step(self, batch, batch_idx):

        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("training_loss", loss, on_step=True)
        self.log("training_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("training_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("validation_loss", loss, on_step=True)
        self.log("validation_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("validation_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("test_loss", loss, on_step=True)
        self.log("test_img_acc", img_acc, on_step=True, prog_bar=True)
        self.log("test_cap_acc", cap_acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        vision_params = {"params": self.vision_encoder.projection.parameters(), "lr": self.lr}
        caption_params = {"params": self.caption_encoder.projection.parameters(), "lr": self.lr}
        return torch.optim.Adam([vision_params, caption_params])


if __name__ == '__main__':

    """
    ve = VisionEncoder()
    dog_img_path = r"https://unsplash.com/photos/QtxgNsmJQSs/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjM1ODQ0MjY3&w=640"
    img = utils.load_image(dog_img_path)
    vision_embeds = ve(img)
    print(vision_embeds)
    """

    te = TextEncoder()
    text = "Hello world"
    text_embeds = te(text)
    print(text_embeds)
