import re
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

from utils import helpers
from model import TopicNetClassifier


# Command-line arguments.
parser = argparse.ArgumentParser(description='Train a TopicNet model.')

parser.add_argument(
    '--train_file', required=True, help='train file path')

parser.add_argument(
    '--valid_split', default=.2, type=float, help='split for validation set')

parser.add_argument(
    '--random_seed', default=1234, type=int, help='random seed for splitting')

parser.add_argument(
    '--token_regex', default='\w+', help='tokenizing regex')

parser.add_argument(
    '--min_freq', default=5, type=int, help='min frequency for vocab')

parser.add_argument(
    '--num_epochs', default=10, type=int, help='number of epochs')

parser.add_argument(
    '--batch_size', default=128, type=int, help='batch size')

parser.add_argument(
    '--learning_rate', default=1e-3, type=float, help='learning rate')

parser.add_argument(
    '--gradient_clipping', default=5.0, type=float, help='gradient clipping')

parser.add_argument(
    '--print_every', default=100, type=int, help='print every n iterations')

parser.add_argument(
    '--num_topics', default=16, type=int, help='number of topics')

parser.add_argument(
    '--num_topic_filters', default=32, type=int, help='number of topic filters')

parser.add_argument(
    '--num_shared_filters', default=16, type=int, help='number of shared filters')

parser.add_argument(
    '--num_dense_layers', default=3, type=int, help='number of dense layers')

parser.add_argument(
    '--filter_size', default=3, type=int, help='filter size')

parser.add_argument(
    '--growth_rate', default=8, type=int, help='growth rate')

parser.add_argument(
    '--disable_cuda', action='store_true', help='disable cuda')

parser.add_argument(
    '--device_id', default=0, type=int, help='id for cuda device')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


# Training function.
def train():
    # Logger.
    logger = helpers.get_logger('training')

    helpers.log_args(logger, args)

    # Prepare training and validation data.
    WORD = re.compile(args.token_regex)

    TEXT = data.Field(lower=True,
                      tokenize=WORD.findall,
                      batch_first=True)

    LABEL = data.Field(sequential=False)

    fields = [('label', LABEL), ('text', TEXT)]

    train_set = data.TabularDataset(args.train_file, 'csv', fields)

    logger.info(f'Loaded training data: {args.train_file}')

    TEXT.build_vocab(train_set,
                     min_freq=args.min_freq)

    LABEL.build_vocab(train_set)

    train_set, valid_set = helpers.split_data(train_set,
                                              fields,
                                              args.random_seed,
                                              args.valid_split)

    logger.info(f'Number of training examples: {len(train_set.examples)}')
    logger.info(f'Number of validation examples: {len(valid_set.examples)}')
    logger.info(f'Size of vocabulary: {len(TEXT.vocab)}')
    logger.info(f'Number of labels: {len(LABEL.vocab)}')

    # Initialize classifier and criterion.
    classifier = TopicNetClassifier(len(TEXT.vocab),
                                    len(LABEL.vocab),
                                    args.num_topics,
                                    args.num_topic_filters,
                                    args.num_shared_filters,
                                    args.num_dense_layers,
                                    args.filter_size,
                                    args.growth_rate)

    if args.cuda:
        classifier.cuda(device=args.device_id)

    criterion = nn.NLLLoss()

    # Training.
    iterator = data.BucketIterator(train_set,
                                   args.batch_size,
                                   lambda x: len(x.text),
                                   device=args.device_id if args.cuda else -1)

    last_epoch = 0
    gamma = (1e-5 / args.learning_rate)**(1 / args.num_epochs)
    learning_rate = args.learning_rate
    for batch in iterator:
        classifier.zero_grad()
        loss = criterion(classifier(batch.text), batch.label)
        loss.backward()

        # Clip gradient and make an SGD step.
        torch.nn.utils.clip_grad_norm(classifier.parameters(), args.gradient_clipping)
        for param in classifier.parameters():
            param.data.add_(-learning_rate, param.grad.data)

        progress, epoch = math.modf(iterator.epoch)

        if iterator.iterations % args.print_every == 0:
            logger.info(f'Epoch {int(epoch):2} | '
                        f'progress: {progress:<6.2%} | '
                        f'loss: {loss.data[0]:6.4f}')

        if progress == 0 and epoch > 0:
            valid_loss, accuracy = helpers.evaluate(valid_set,
                                                    args.batch_size,
                                                    classifier,
                                                    args.device_id if args.cuda else -1)

            logger.info(f'Validation accuracy: {accuracy:<6.2%}')
            logger.info(f'Average validation loss: {valid_loss:6.4f}')
            classifier.train()

        # Decay learning rate for each new epoch.
        if epoch > last_epoch:
            last_epoch = epoch
            learning_rate *= gamma

        if epoch == args.num_epochs:
            break


if __name__ == '__main__':
    train()
