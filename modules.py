import torch
import torch.nn as nn


class TopicLayer(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_topics,
                 num_topic_filters,
                 num_shared_filters):

        super(TopicLayer, self).__init__()

        self.shared_filters = num_shared_filters > 0

        self.topic_embeddings = nn.ModuleList()
        for i in range(num_topics):
            self.topic_embeddings.append(nn.Embedding(vocab_size, num_topic_filters))

        if self.shared_filters:
            self.shared_embedding = nn.Embedding(vocab_size, num_shared_filters)

    def forward(self, sequence):
        topics = []
        for topic_embedding in self.topic_embeddings:
            topics.append(topic_embedding(sequence).transpose(1, 2))

        if self.shared_filters:
            shared_topic = self.shared_embedding(sequence).transpose(1, 2)
            topics = [torch.cat([topic, shared_topic], 1) for topic in topics]

        return topics


class DenseLayer(nn.Sequential):
    def __init__(self,
                 num_input_filters,
                 growth_rate,
                 filter_size,
                 padding):

        super(DenseLayer, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_filters))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(num_input_filters,
                                          growth_rate,
                                          (1, filter_size),
                                          padding=padding))

    def forward(self, feature_maps):
        new_feature_maps = super(DenseLayer, self).forward(feature_maps)
        return torch.cat([feature_maps, new_feature_maps], 1)


class DenseNet(nn.Module):
    def __init__(self,
                 num_input_filters,
                 num_layers,
                 growth_rate,
                 filter_size):

        super(DenseNet, self).__init__()

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(f'dense-{i}',
                                   DenseLayer(num_input_filters,
                                              growth_rate,
                                              filter_size,
                                              padding=(0, (filter_size - 1) // 2)))
            num_input_filters += growth_rate

        self.layers.add_module('activation', nn.ReLU())

    def forward(self, topic):
        topic = topic.unsqueeze(2)
        feature_maps = self.layers(topic)
        return feature_maps.squeeze(2).max(2)[0]
