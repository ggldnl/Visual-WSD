import torch.nn.functional as F
import torch


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(sim):
    caption_loss = contrastive_loss(sim, dim=0)
    image_loss = contrastive_loss(sim, dim=0)
    return (caption_loss + image_loss) / 2.0


def similarity(v1, v2):

    # Normalize the embeddings
    embedding1_normalized = F.normalize(v1, p=2, dim=-1)
    embedding2_normalized = F.normalize(v2, p=2, dim=-1)

    # Compute and return cosine similarity between the two vectors
    return F.cosine_similarity(embedding1_normalized, embedding2_normalized, dim=-1)


def metrics(sim):

    # TODO

    y = torch.arange(len(sim))
    img2cap_match_idx = sim.argmax(dim=1)
    cap2img_match_idx = sim.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc
