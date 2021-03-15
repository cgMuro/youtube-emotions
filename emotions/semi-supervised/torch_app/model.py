import torch.nn as nn
from transformers import BertModel

N_EMOTIONS = 5

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()

        # Bert model
        # self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=N_EMOTIONS)
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        
        # Feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, N_EMOTIONS)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token
        cls = outputs[0][:, 0, :]
        # Classify
        logits = self.classifier(cls)

        return logits
        