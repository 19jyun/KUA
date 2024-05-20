from transformers import RobertaForSequenceClassification

#Last version before merging into multimodal

def get_model(model_name='roberta-base', num_labels=2):
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
