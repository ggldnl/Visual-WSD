from transformers import ViTImageProcessor, DistilBertModel, DistilBertTokenizer, ViTModel
from torchvision import transforms as tt
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import functions
import config
import torch


class Projection(pl.LightningModule):

    def __init__(self, d_in, d_hidden, d_out, p=0.5):

        super().__init__()

        self.linear1 = nn.Linear(d_in, d_hidden, bias=False)
        self.linear2 = nn.Linear(d_hidden, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds
        """

        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.layer_norm(x)
        return x


class VisionEncoder(pl.LightningModule):

    def __init__(self, d_hidden, d_out):

        super().__init__()

        # Create the vision transformer
        self.preprocessor = ViTImageProcessor(config.VISION_MODEL)
        self.vision_model = ViTModel.from_pretrained(config.VISION_MODEL)

        # Take in and out dimensions of the last linear layer
        last_linear_layer = self.vision_model.encoder.layer[-1].output.dense
        last_linear_layer_out = last_linear_layer.out_features

        if d_hidden is None:
            d_hidden = last_linear_layer_out

        # Create a projection layer using the size of the vision models output features as in dimension
        self.projection = Projection(last_linear_layer_out, d_hidden, d_out)

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

    def __init__(self, d_hidden, d_out):

        super().__init__()

        # Create the text transformer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.TEXT_MODEL)
        self.text_model = DistilBertModel.from_pretrained(config.TEXT_MODEL)

        # Take in and out dimensions of the last linear layer
        last_linear_layer = self.text_model.transformer.layer[-1].output_layer_norm
        last_linear_layer_out = last_linear_layer.normalized_shape[0]

        if d_hidden is None:
            d_hidden = last_linear_layer_out

        # Create a projection layer using the size of the vision models output features as in dimension
        self.projection = Projection(last_linear_layer_out, d_hidden, d_out)

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

    def __init__(self,
                 img_proj_hidden_size=None,
                 text_proj_hidden_size=None,
                 embed_dim=config.EMBED_DIM,
                 lr=config.LEARNING_RATE
                 ):

        super().__init__()

        self.vision_encoder = VisionEncoder(img_proj_hidden_size, embed_dim)
        self.caption_encoder = TextEncoder(text_proj_hidden_size, embed_dim)
        self.lr = lr

    def common_step(self, batch):

        # Take images and text from the batch
        images, texts = batch

        # Create the embeddings
        image_embeds = self.vision_encoder(images)
        text_embeds = self.caption_encoder(texts)

        # Compute similarity and loss
        sim = functions.similarity_matrix(image_embeds, text_embeds)
        loss = functions.clip_loss(sim)
        img_acc, cap_acc = functions.clip_metrics(sim)

        return loss, img_acc, cap_acc

    def training_step(self, batch, batch_idx):

        loss, img_acc, cap_acc = self.common_step(batch)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("validation_loss", loss)
        self.log("validation_img_acc", img_acc, prog_bar=True)
        self.log("validation_cap_acc", cap_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        loss, img_acc, cap_acc = self.common_step(batch)
        self.log("test_loss", loss)
        self.log("test_img_acc", img_acc, prog_bar=True)
        self.log("test_cap_acc", cap_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # vision_params = {"params": self.vision_encoder.projection.parameters(), "lr": self.lr}
        # caption_params = {"params": self.caption_encoder.projection.parameters(), "lr": self.lr}
        # Specify parameters to optimize

        trainable_parameters = list(self.caption_encoder.projection.parameters()) + \
                               list(self.vision_encoder.projection.parameters())

        # Define optimizer (e.g., Adam optimizer)
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr)

        return optimizer

    def top_k_images(self, sentence, images, k=1, image_paths=None):
        """
        Returns the k most similar images to the sentence among the provided ones.
        If the image_paths are provided, the method will automatically return the
        paths of the k most similar images instead of the images themselves.
        """

        assert len(images) > k, f"Can't return {k} image(s) if {len(images)} provided"

        image_transform = tt.Compose([
            tt.Resize((255, 255)),
            tt.ToTensor()
        ])
        images_tensors = [image_transform(image) for image in images]

        # Create the embeddings
        image_embeds = self.vision_encoder(images_tensors)
        sentence_embed = self.caption_encoder(sentence)

        similarities = [functions.cosine_similarity(sentence_embed, image_embed).item() for image_embed in image_embeds]

        # Get indices of top k images
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        if image_paths is None:

            # Return the images
            return [images[i] for i in top_k_indices]

        # Return their paths instead
        return [image_paths[i] for i in top_k_indices]

    def top_k_texts(self, image, sentences, k=1):
        """
        Returns the k most similar sentences to the image among the provided ones.
        """

        assert len(sentences) > k, f"Can't return {k} sentences(s) if {len(sentences)} provided"

        # Create the embeddings
        image_embed = self.vision_encoder(image)
        sentence_embeds = self.caption_encoder(sentences)

        similarities = [functions.cosine_similarity(sentence_embed, image_embed).item() for sentence_embed in sentence_embeds]

        # Get indices of top k sentences
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Get the actual top k sentences
        top_k_sentences = [sentences[i] for i in top_k_indices]

        return top_k_sentences


if __name__ == '__main__':

    import utils
    ve = VisionEncoder(None, config.EMBED_DIM)
    dog_img_path = r"https://unsplash.com/photos/QtxgNsmJQSs/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjM1ODQ0MjY3&w=640"
    img = utils.load_image(dog_img_path)
    vision_embeds = ve(img)
    print(vision_embeds)

    """
    te = TextEncoder(None, config.EMBED_DIM)
    text = "Hello world"
    text_embeds = te(text)
    print(text_embeds)
    """
