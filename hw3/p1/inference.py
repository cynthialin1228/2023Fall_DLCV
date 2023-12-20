import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import clip
import argparse

class ImageDataset(Dataset):
    def __init__(self, data_directory):
        self.image_paths = sorted([
            os.path.join(data_directory, filename)
            for filename in os.listdir(data_directory)
            if filename.lower().endswith('.png')
        ])
        self.image_filenames = sorted([
            filename
            for filename in os.listdir(data_directory)
            if filename.lower().endswith('.png')
        ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image_tensor = TF.to_tensor(image)
        label = self.image_filenames[index].split("_")[0]
        return image_tensor, label, self.image_filenames[index]

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument("--img_dir", "-i", type=str, default="./", help="img_dir")
    parser.add_argument("--json_path", "-j", type=str, default="./id2label.json", help="json_path")
    parser.add_argument("--output", "-o", type=str, default="./", help="output")

    return parser.parse_args()

def infer(img_dir, json_path, output):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load('ViT-B/16', device_type)
    image_dataset = ImageDataset(img_dir)
    image_loader = DataLoader(image_dataset, batch_size=1, shuffle=True, pin_memory=True)
    with open(json_path) as file:
        label_classes = json.load(file)
    prediction_count = 0
    predictions_df = pd.DataFrame({"filename": [], "label": []})
    for image_tensor, label, filename in tqdm(image_loader, position=0, leave=True):
        # Prepare image input for CLIP
        pil_image = TF.to_pil_image(image_tensor.squeeze(0))
        clip_image_input = clip_preprocess(pil_image).unsqueeze(0).to(device_type)

        # Prepare text inputs for CLIP model
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {class_name}") for _, class_name in label_classes.items()]).to(device_type)

        # Calculate features and similarity
        with torch.no_grad():
            image_features = clip_model.encode_image(clip_image_input)
            text_features = clip_model.encode_text(text_inputs)

        # Normalize and find top label
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_values, top_indices = similarity[0].topk(1)
        top_indices = top_indices.detach().cpu().numpy().astype(str)

        # Add prediction to DataFrame
        filename_str = filename[0] if isinstance(filename, list) else filename
        # prediction_row = pd.DataFrame({"filename": [filename], "label": [top_indices[0]]})
        prediction_row = pd.DataFrame({"filename": filename, "label": top_indices[0]}, index=[0])
        predictions_df = pd.concat([predictions_df, prediction_row])

    # Save predictions to CSV
    predictions_df.to_csv(output, index=False)

if __name__ == "__main__":
    args = get_args()
    infer(args.img_dir, args.json_path, args.output)