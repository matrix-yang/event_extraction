"""
@Time : 2021/4/159:31
@Auth : 周俊贤
@File ：model.py
@DESCRIPTION:

"""
from torch import nn
from transformers import BertModel
from torchcrf import CRF

class DuEEEvent_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuEEEvent_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(sequence_output)
        return logits

class DuEECls_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuEECls_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(pooled_output)
        return logits


class DuEEEvent_crf_model(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(DuEEEvent_crf_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.to(torch.uint8))
            return -1 * loss
        return logits

if __name__ == '__main__':
    model = DuEEEvent_model("/data/zhoujx/prev_trained_model/rbt3", num_classes=60)
    a = 1