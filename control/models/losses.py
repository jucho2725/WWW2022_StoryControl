"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from enum import Enum
import torch.nn.functional as F
import torch.distributed as dist

def align_loss(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    x = F.normalize(x, p=2, dim=1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class DistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)



class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:
    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).
    Margin is an important hyperparameter and needs to be tuned respectively.
    For further details, see: https://en.wikipedia.org/wiki/Triplet_loss
    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.
    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1', 'Negative 1']),
            InputExample(texts=['Anchor 2', 'Positive 2', 'Negative 2'])]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.TripletLoss(model=model)
    """
    def __init__(self, device, in_batch_supervision=True, distance_metric=DistanceMetric.COSINE_DISTANCE, triplet_margin: float = 5):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin
        self.in_batch_supervision = in_batch_supervision
        self.device = device

    def forward(self, features, labels, neg_labels):

        if self.in_batch_supervision:
            rep_pos, rep_neg = self.make_samples(features, labels, neg_labels)
        else:
            rep_pos, rep_neg = self.make_samples_no_inbatchsup(features, labels, neg_labels)

        rep_pos_anchor, rep_pos_other = rep_pos[:, 0], rep_pos[:, 1]
        rep_neg_anchor, rep_neg_other = rep_neg[:, 0], rep_neg[:, 1]
        distance_pos = self.distance_metric(rep_pos_anchor, rep_pos_other)
        distance_neg = self.distance_metric(rep_neg_anchor, rep_neg_other)
        losses = F.relu(distance_pos.mean() - distance_neg.mean() + self.triplet_margin)
        return losses

    def make_samples(self, features, labels, neg_labels):
        batch_size = features.size()[0]
        reconst_pos = []
        reconst_neg = []

        for i in range(batch_size):
            anchor = features[i][0]  # anchor emb
            positive = features[i][1]  # pos
            negative = features[i][2]  # neg

            reconst_pos.append(torch.stack([anchor, positive]))  # A A' 과 같이 pos 쌍 들어감
            reconst_neg.append(torch.stack([anchor, negative]))

            anchor_label = labels[i]

            for j in range(i + 1, batch_size):
                if anchor_label == labels[j]:
                    reconst_pos.append(torch.stack([anchor, features[j][0]]))
                else:
                    reconst_neg.append(torch.stack([anchor, features[j][0]]))

                if anchor_label == neg_labels[j]:
                    reconst_pos.append(torch.stack([anchor, features[j][2]]))
                else:
                    reconst_neg.append(torch.stack([anchor, features[j][2]]))

        reconst_pos = torch.stack(reconst_pos).to(self.device)
        reconst_neg = torch.stack(reconst_neg).to(self.device)
        return reconst_pos, reconst_neg

    # contrast pos pos' neg - no supervision
    # 자기 자신만 pos 고 나머지는 다 negative
    def make_samples_no_inbatchsup(self, features, labels, neg_labels):
        batch_size = features.size()[0]
        reconst_pos = []
        reconst_neg = []

        for i in range(batch_size):
            anchor = features[i][0]  # anchor emb
            positive = features[i][1]  # pos
            negative = features[i][2]   # neg

            reconst_pos.append(torch.stack([anchor, positive]))  # A A' 과 같이 pos 쌍 들어감
            reconst_neg.append(torch.stack([anchor, negative]))

            for j in range(i + 1, batch_size):
                reconst_neg.append(torch.stack([anchor, features[j][0]]))
                reconst_neg.append(torch.stack([anchor, features[j][2]]))

        reconst_pos = torch.stack(reconst_pos).to(self.device)
        reconst_neg = torch.stack(reconst_neg).to(self.device)
        return reconst_pos, reconst_neg


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.
    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.ContrastiveLoss(model=model)
    """

    def __init__(self, distance_metric=DistanceMetric.COSINE_DISTANCE, margin: float = 0.5, size_average:bool = True, in_batch_supervision = True, device=None):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average
        self.device = device
        self.in_batch_supervision = in_batch_supervision

    def forward(self, features, labels, neg_labels=None):
        if self.in_batch_supervision:
            features, labels = self.make_samples_neg(features, labels, neg_labels)
        else:
            features, labels = self.make_samples_neg_no_inbatch(features, labels, neg_labels)
        rep_anchor = features[:, 0]
        rep_other = features[:, 1]
        distances = self.distance_metric(rep_anchor, rep_other)

        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()
    #

    # contrast pos pos' neg
    def make_samples_neg(self, features, labels, neg_labels):
        batch_size = features.size()[0]
        reconst = []
        re_labels = []

        for i in range(batch_size):
            anchor_pos1 = features[i][0]  # anchor emb
            anchor_pos2 = features[i][1]  # pos
            anchor_neg = features[i][2]   # neg

            anchor_label = labels[i]
            anchor_neg_label = neg_labels[i]

            # pos pos'
            reconst.append(torch.stack([anchor_pos1, anchor_pos2]))  # A A' 과 같이 pos 쌍 들어감
            re_labels.append(1)  # pos

            # pos neg
            reconst.append(torch.stack([anchor_pos2, anchor_neg]))
            reconst.append(torch.stack([anchor_pos1, anchor_neg]))
            if anchor_label == anchor_neg_label:
                re_labels.extend([1, 1])
            else:
                re_labels.extend([0, 0])

            for j in range(i + 1, batch_size):
                # pos and other's pos
                reconst.append(torch.stack([anchor_pos1, features[j][0]]))
                reconst.append(torch.stack([anchor_pos2, features[j][0]]))
                reconst.append(torch.stack([anchor_pos1, features[j][1]]))
                reconst.append(torch.stack([anchor_pos2, features[j][1]]))
                if anchor_label == labels[j]:
                    re_labels.extend([1, 1, 1, 1])
                else:
                    re_labels.extend([0, 0, 0, 0])

                # neg and other's pos
                reconst.append(torch.stack([anchor_neg, features[j][0]]))
                reconst.append(torch.stack([anchor_neg, features[j][1]]))
                if anchor_neg_label == labels[j]:
                    re_labels.extend([1, 1])
                else:
                    re_labels.extend([0, 0])

                # pos and other's neg
                reconst.append(torch.stack([anchor_pos1, features[j][2]]))
                reconst.append(torch.stack([anchor_pos2, features[j][2]]))
                if anchor_label == neg_labels[j]:
                    re_labels.extend([1, 1])
                else:
                    re_labels.extend([0, 0])

                # neg and other's neg
                reconst.append(torch.stack([anchor_neg, features[j][2]]))
                if anchor_neg_label == neg_labels[j]:
                    re_labels.append(1)
                else:
                    re_labels.append(0)

        reconst = torch.stack(reconst).to(self.device)
        labels = torch.tensor(re_labels, dtype=torch.float).to(self.device)
        return reconst, labels

    # contrast pos pos' neg - no supervision
    # 자기 자신만 pos 고 나머지는 다 negative
    def make_samples_neg_no_inbatch(self, features, labels, neg_labels):
        batch_size = features.size()[0]
        reconst = []
        re_labels = []

        for i in range(batch_size):
            anchor_pos1 = features[i][0]  # anchor emb
            anchor_pos2 = features[i][1]  # pos
            anchor_neg = features[i][2]   # neg

            reconst.append(torch.stack([anchor_pos1, anchor_pos2]))  # A A' 과 같이 pos 쌍 들어감
            re_labels.append(1)  # pos

            reconst.append(torch.stack([anchor_pos2, anchor_neg]))
            reconst.append(torch.stack([anchor_pos1, anchor_neg]))
            re_labels.extend([0, 0])

            for j in range(i + 1, batch_size):
                reconst.append(torch.stack([anchor_pos1, features[j][0]]))
                reconst.append(torch.stack([anchor_pos2, features[j][0]]))
                reconst.append(torch.stack([anchor_pos1, features[j][1]]))
                reconst.append(torch.stack([anchor_pos2, features[j][1]]))
                re_labels.extend([0, 0, 0, 0])

        reconst = torch.stack(reconst).to(self.device)
        labels = torch.tensor(re_labels, dtype=torch.float).to(self.device)
        return reconst, labels


class CELoss(nn.Module):
    def __init__(self, temp, device, in_batch_supervision):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device
        self.in_batch_supervision = in_batch_supervision

    def sim(self, x, y):
        return self.cos(x, y) / self.temp

    def forward(self, features, labels, neg_labels):
        batch_size = features.size()[0]

        if self.in_batch_supervision:
            # build label info matrix (for multi-label classification)
            pos_stacks = labels.repeat(batch_size, 1)
            pospos_labels = torch.eq(labels.reshape(batch_size, 1), pos_stacks) # [bsz, bsz]
            neg_stacks = neg_labels.repeat(batch_size, 1)
            posneg_labels = torch.eq(labels.reshape(batch_size, 1), neg_stacks) # [bsz. bsz]
            labels = torch.cat([pospos_labels, posneg_labels], 1).float().to(self.device) # [bsz, 2 * bsz]
            loss_fct = nn.BCEWithLogitsLoss()
        else:
            labels = torch.arange(batch_size).long().to(self.device)
            #
            loss_fct = nn.CrossEntropyLoss()


        # Separate representation
        z1, z2 = features[:,0], features[:,1]
        # Hard negative
        num_sent = features.size(1)
        if num_sent == 3:
            z3 = features[:, 2]

        # Gather all embeddings if using distributed training
        if dist.is_initialized():
            # Gather hard negative
            if num_sent >= 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)



        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            # z3_weight = self.model_args.hard_negative_weight
            z3_weight = 1
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(self.device)

            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)

        return loss