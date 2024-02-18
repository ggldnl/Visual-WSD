import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch


"""
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


def similarity_matrix(v1, v2):
    return v1 @ v2.T


def metrics(sim):

    y = torch.arange(len(sim))
    img2cap_match_idx = sim.argmax(dim=1)
    cap2img_match_idx = sim.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc
"""


def cosine_similarity(v1, v2):

    # Normalize the embeddings
    embedding1_normalized = F.normalize(v1, p=2, dim=-1)
    embedding2_normalized = F.normalize(v2, p=2, dim=-1)

    # Compute and return cosine similarity between the two vectors
    return F.cosine_similarity(embedding1_normalized, embedding2_normalized, dim=-1)


def similarity_matrix(image_embeddings, text_embeddings, temperature=0.07):

    # Normalize batches across embeddings dimension
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Compute cosine similarity matrix
    sim_matrix = torch.matmul(image_embeddings, text_embeddings.t()) / temperature

    return sim_matrix


def plot_similarity_matrix(sim_matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Text Embeddings')
    plt.ylabel('Image Embeddings')
    plt.title('Similarity Matrix')
    plt.show()


def clip_loss(sim_matrix):

    # Compute positive and negative examples
    batch_size = sim_matrix.size(0)
    positives = torch.diag(sim_matrix).reshape(batch_size, 1)
    negatives = torch.exp(sim_matrix - torch.eye(batch_size).to(sim_matrix.device))

    # Compute Negative Log Likelihood Loss
    numerator = torch.exp(positives)
    denominator = torch.exp(positives) + torch.sum(negatives, dim=1)
    loss = -torch.mean(torch.log(numerator / denominator))

    return loss


def clip_metrics(sim_matrix):

    # Get the number of samples
    num_samples = sim_matrix.shape[0]

    target = torch.arange(num_samples)

    # For each image, find the index of the most similar caption
    image_predictions = torch.argmax(sim_matrix, axis=1)

    # For each caption, find the index of the most similar image
    caption_predictions = torch.argmax(sim_matrix, axis=0)

    # Compute image accuracy
    correct_image_predictions = image_predictions == target
    image_accuracy = torch.sum(correct_image_predictions).item() / num_samples

    # Compute caption accuracy
    correct_caption_predictions = caption_predictions == target
    caption_accuracy = torch.sum(correct_caption_predictions).item() / num_samples

    return image_accuracy, caption_accuracy


if __name__ == '__main__':

    # Try the clip loss with two randomly generated batches
    batch1 = torch.randn(64, 512)
    batch2 = torch.randn(64, 512)
    batches = [
        batch2.detach().clone(),
        batch1.detach().clone(),
        batch2.detach().clone() + torch.randn_like(batch2) * 2  # Add random noise to the tensor
    ]
    descriptions = [
        'random batches',
        'identical batches',
        'similar batches'
    ]

    for batch, description in zip(batches, descriptions):
        sim = similarity_matrix(batch1, batch)
        loss = clip_loss(sim)
        img_acc, cap_acc = clip_metrics(sim)
        print(f'CLIP with {description}:')
        print(f'loss            : {loss}')
        print(f'image accuracy  : {img_acc}')
        print(f'caption accuracy: {cap_acc}')
        print('-' * 50)
