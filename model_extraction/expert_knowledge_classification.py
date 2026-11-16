import os
import torch
import random
import numpy as np
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    BartConfig,
    BartModel,
    BartPreTrainedModel
)

from transformers.modeling_outputs import SequenceClassifierOutput


# =====================================================
# SEED
# =====================================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "PLM_classification/bart_123_final_model"

NUM_LABELS = 6
MAX_LENGTH = 256

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


LABEL_NAMES = {
    0: "state_deviation",
    1: "process_flow",
    2: "maintenance",
    3: "operational",
    4: "structural",
    5: "temporal_and_probabilistic",
}


# =====================================================
# CUSTOM BART MODEL
# =====================================================

class BartForMultiLabelClassification(
    BartPreTrainedModel
):

    def __init__(self, config):

        super().__init__(config)

        self.num_labels = NUM_LABELS

        self.model = BartModel(config)

        self.dropout = nn.Dropout(
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.dropout
        )

        self.classifier = nn.Linear(
            config.d_model,
            config.num_labels
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.post_init()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden = outputs.last_hidden_state


        mask = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )


        pooled = (
            hidden * mask
        ).sum(dim=1)


        pooled = pooled / (
            mask.sum(dim=1)
            .clamp(min=1e-9)
        )


        pooled = self.dropout(
            pooled
        )


        logits = self.classifier(
            pooled
        )


        return SequenceClassifierOutput(
            logits=logits
        )


# =====================================================
# LOAD MODEL ONCE
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


config = BartConfig.from_pretrained(
    MODEL_PATH
)

config.num_labels = NUM_LABELS


model = BartForMultiLabelClassification.from_pretrained(
    MODEL_PATH,
    config=config
)


model.to(device)

model.eval()


# =====================================================
# CLASSIFICATION FUNCTION
# KEEP ORIGINAL INPUT FORMAT
# =====================================================

def classify_statement(original_input):

    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    label_names = {
        0: "state_deviation",
        1: "process_flow",
        2: "maintenance",
        3: "operational",
        4: "structural",
        5: "temporal_and_probabilistic",
    }


    inputs = tokenizer(
        original_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )


    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    probabilities = torch.sigmoid(
        outputs.logits
    )


    # Same behavior as your original code:
    # select label with highest confidence
    pred = torch.argmax(
        probabilities,
        dim=1
    ).item()


    confidence = probabilities[0][pred].item()


    pred_label = label_names[pred]


    print("################# pred index:", pred)
    print("################# label:", pred_label)
    print("################# confidence:", confidence)


    print("################# All confidences:")

    for i, prob in enumerate(probabilities[0]):

        print(
            f"{label_names[i]:30s}: {prob.item():.4f}"
        )


    return pred_label
