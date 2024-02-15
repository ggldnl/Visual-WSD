import torch.nn.functional as F
import torch


def contrastive_loss(logits, dim):
    # Contrastive loss helps the model learn to pull similar images closer
    # in the embedding space while pushing dissimilar images farther apart.
    #
    # Say we have 3 data points pairs (img_point_1, text_point_1),
    # (img_point_2, text_point_2), (img_point_3, text_point_3) (a batch)
    # We compute the similarity scores for each img_point with respect to each
    # text_point. The similarity scores of the original pairs will be on the
    # diagonal. We take the log_softmax converts the logits to probabilities
    # (they sum up to 1). The mean returns a single loss value that is
    # representing the average negative log probability of the correct classes
    # across all pairs of data points. It is negative since we usually want to
    # minimize this value.
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(sim):
    caption_loss = contrastive_loss(sim, dim=0)
    image_loss = contrastive_loss(sim, dim=1)
    return (caption_loss + image_loss) / 2.0


def cosine_similarity(v1, v2):

    # Normalize the embeddings
    embedding1_normalized = F.normalize(v1, p=2, dim=-1)
    embedding2_normalized = F.normalize(v2, p=2, dim=-1)

    # Compute and return cosine similarity between the two vectors
    return F.cosine_similarity(embedding1_normalized, embedding2_normalized, dim=-1)


def similarity_matrix(v1, v2):
    return v1 @ v2.T


def metrics(sim):

    y = torch.arange(len(sim))
    img2cap_match_idx = sim.argmax(dim=1)
    cap2img_match_idx = sim.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc
