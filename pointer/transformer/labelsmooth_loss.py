import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmothingLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.smoothing_value = config.ls / (config.vocab_size - 2)
        self.one_hot = torch.full((config.vocab_size, ),
                                  self.smoothing_value)
        self.confidence = 1.0 - config.ls
        self.pad = config.pad_id
        self.vocab_size = config.vocab_size

    def forward(self, output, target):
        """
        :param output: (batch_size, vocab_size)
        :param target: (batch_size, len)
        """
        target = target.view(-1)
        output = output.view(-1, self.vocab_size)
        model_prob = self.one_hot.repeat(target.size(0),
                                         1)  # (batch, vocab_size)
        if torch.cuda.is_available():
            model_prob = model_prob.cuda()
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.pad).unsqueeze(1), 0)

        output = torch.nn.functional.log_softmax(output, dim=-1)

        return F.kl_div(output, model_prob, reduction='sum')