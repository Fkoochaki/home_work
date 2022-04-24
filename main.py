import pdb
import sys
import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn

from transformers import ViTFeatureExtractor, ViTModel, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, load_metric
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

from dataset import Dataset


class ViTClassification(nn.Module):
    """
    Creating a binary VisionTransformer model.
    """
    def __init__(self, num_labels=2):
        super(ViTClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def my_collate_fn(examples):
    """
    Data collector for trainer.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    """
    Computing the accuracy.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)


def train():
    # Loading the ViTFeatureExtractor and get its normalization statistic
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    train_transforms = Compose(
        [
            #RandomResizedCrop(feature_extractor.size),
            Resize((224, 224)),
            #RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ]
    )

    val_transforms = Compose(
        [
            Resize((224, 224)),
            #CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize
        ]
    )

    train_ds = Dataset("train", transform=train_transforms) # Train dataset
    val_ds = Dataset("val", transform=val_transforms) # Validation dataset
    
    model = ViTClassification()

    num_epoch = 6

    # Training parameters
    args = TrainingArguments(
            f"intuitive_hm",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8, # I have only access to one GPU
            per_device_eval_batch_size=4,
            num_train_epochs=num_epoch,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='logs',
            remove_unused_columns=False,
        )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=my_collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    # Train the model
    trainer.train()

    # Saving the last model
    torch.save({'model_state_dict': model.state_dict()}, "./intuitive_hm.pt")


def inference(vid_path):
    # Loading the model
    device = "cpu"
    model = ViTClassification()
    checkpoint = torch.load("./intuitive_hm.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model is loaded!")

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    
    val_transforms = Compose(
        [
            Resize((224, 224)),
            #CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize
        ]
    )
    
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()

    count = 0
    while success:
        # Converting opencv image to pillow image type
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil = val_transforms(image_pil)
        inputs = feature_extractor(image_pil, return_tensors="pt")

        with torch.no_grad():
            logits = model(inputs["pixel_values"], labels=None).logits
            predicted_label = logits.argmax(-1).item()
        print("Frame: {}, Pred_lbl: {}".format(count, predicted_label))
        
        # Read the next frame
        success,image = vidcap.read()
        count += 1


if __name__ == "__main__":

    if sys.argv[1] == "train": # Training
        train()
    elif sys.argv[1] == "infer": # Inference
        name = "caseid_000001_fps1.mp4"
        inference(name)
