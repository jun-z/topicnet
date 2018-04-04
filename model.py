import torch
import torch.nn as nn

from modules import TopicLayer, DenseNet


class TopicNetClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 labelset_size,
                 num_topics,
                 num_topic_filters,
                 num_shared_filters,
                 num_dense_layers,
                 filter_size,
                 growth_rate,
                 dropout_prob):

        super(TopicNetClassifier, self).__init__()

        self.num_filters = num_topic_filters + num_shared_filters

        self.topic_layer = TopicLayer(vocab_size,
                                      num_topics,
                                      num_topic_filters,
                                      num_shared_filters)

        self.dense_nets = nn.ModuleList()
        for i in range(num_topics):
            self.dense_nets.append(DenseNet(self.num_filters,
                                            num_dense_layers,
                                            growth_rate=growth_rate,
                                            filter_size=filter_size))

        self.num_filters += num_dense_layers * growth_rate

        self.projs = nn.ModuleList()
        for i in range(num_topics):
            self.projs.append(nn.Linear(self.num_filters, labelset_size))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        topics = self.topic_layer(sequence)

        feature_vecs = []
        for topic, dense_net in zip(topics, self.dense_nets):
            feature_vecs.append(dense_net(topic))

        all_logits = []
        for feature_vec, proj in zip(feature_vecs, self.projs):
            all_logits.append(proj(feature_vec))

        feature_vecs = torch.cat([vec.unsqueeze(1) for vec in feature_vecs], 1)
        all_logits = torch.cat([logits.unsqueeze(1) for logits in all_logits], 1)

        batch_size = feature_vecs.shape[0]
        _, max_indices = feature_vecs.mean(-1).max(-1)

        logits = all_logits[range(batch_size), max_indices]
        return self.softmax(logits)
