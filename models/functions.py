import torch.nn.functional as F
import torch


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(similarity: torch.Tensor):
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0


def metrics(similarity: torch.Tensor):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


def similarity_function(self, embedding1, embedding2):
    # Normalize the embeddings
    embedding1_normalized = F.normalize(embedding1, p=2, dim=-1)
    embedding2_normalized = F.normalize(embedding2, p=2, dim=-1)

    # Calculate cosine similarity
    similarity = F.cosine_similarity(embedding1_normalized, embedding2_normalized, dim=-1)

    return similarity
