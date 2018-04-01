import torch
import torch.nn as nn


class TopicLayer(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_topics,
                 num_topic_filters,
                 num_shared_filters):

        super(TopicLayer, self).__init__()

        self.shared_filters = num_shared_filters > 0

        self.topic_convs = nn.ModuleList()
        for i in range(num_topics):
            self.topic_convs.append(nn.Conv2d(1,
                                              num_topic_filters,
                                              (1, embedding_dim)))

        if self.shared_filters:
            self.shared_conv = nn.Conv2d(1,
                                         num_shared_filters,
                                         (1, embedding_dim))

    def forward(self, embeddings):
        embeddings = embeddings.unsqueeze(1)

        topics = []
        for conv in self.topic_convs:
            topics.append(conv(embeddings).squeeze(-1))

        if self.shared_filters:
            shared_topic = self.shared_conv(embeddings).squeeze(-1)
            topics = [torch.cat([topic, shared_topic], 1) for topic in topics]

        return topics


class DenseLayer(nn.Sequential):
    def __init__(self,
                 num_filters,
                 filter_size,
                 growth_rate,
                 padding):

        super(DenseLayer, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_filters))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv2d(num_filters,
                                          growth_rate,
                                          (1, filter_size),
                                          padding=padding))

    def forward(self, feature_maps):
        new_feature_maps = super(DenseLayer, self).forward(feature_maps)
        return torch.cat([feature_maps, new_feature_maps], 1)


class DenseNet(nn.Module):
    def __init__(self,
                 num_layers,
                 num_filters,
                 filter_size,
                 growth_rate):

        super(DenseNet, self).__init__()

        self.padding = (0, max(0, (filter_size - 1) // 2))

        self.layers = nn.Sequential()
        self.layers.add_module('norm-0', nn.BatchNorm2d(1))
        self.layers.add_module('relu-0', nn.ReLU())
        self.layers.add_module('conv-0', nn.Conv2d(1,
                                                   growth_rate,
                                                   (num_filters, filter_size),
                                                   padding=self.padding))

        num_feature_maps = growth_rate
        for i in range(num_layers):
            self.layers.add_module(f'dense-{i}',
                                   DenseLayer(num_feature_maps,
                                              filter_size,
                                              growth_rate,
                                              padding=self.padding))
            num_feature_maps += growth_rate

        self.layers.add_module('norm-1', nn.BatchNorm2d(num_feature_maps))
        self.layers.add_module('relu-1', nn.ReLU())
        self.layers.add_module('conv-1', nn.Conv2d(num_feature_maps,
                                                   num_layers * growth_rate,
                                                   (1, filter_size),
                                                   padding=self.padding))
        self.layers.add_module('relu-2', nn.ReLU())

    def forward(self, topic):
        topic = topic.unsqueeze(1)
        feature_maps = self.layers(topic).squeeze(2)
        return feature_maps.max(2)[0]
