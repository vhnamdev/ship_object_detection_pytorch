import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ship_dataloader import get_data_loaders
from ship_model import get_model

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(area1 + area2 - intersection_area)

def train():
    NUM_CLASSES = 7
    DATA_DIR = 'data/seaship.coco'
    NUM_EPOCHS = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    history = {
        'train_loss': [],
        'valid_loss': []
    }

    train_loader, valid_loader, test_loader = get_data_loaders(data_dir=DATA_DIR, batch_size=4)
    model = get_model(num_classes=NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_epoch_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, targets in train_pbar:
            images = list(image.to(device) for image in images)
            targets = list({k: v.to(device) for k, v in t.items()} for t in targets)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_epoch_loss += losses.item()
            train_pbar.set_postfix(loss=losses.item())

        history['train_loss'].append(train_epoch_loss / len(train_loader))

        valid_epoch_loss = 0
        model.train() 
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]")
            for images, targets in valid_pbar:
                images = list(image.to(device) for image in images)
                targets = list({k: v.to(device) for k, v in t.items()} for t in targets)

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                valid_epoch_loss += losses.item()
                valid_pbar.set_postfix(v_loss=losses.item())
        
        history['valid_loss'].append(valid_epoch_loss / len(valid_loader))
        print(f"Epoch {epoch+1} Summary: Train Loss: {history['train_loss'][-1]:.4f} | Valid Loss: {history['valid_loss'][-1]:.4f}")

    print("\nStarting Evaluation on Test Set...")
    model.eval()
    correct_count = 0
    wrong_count = 0
    iou_threshold = 0.5 
    score_threshold = 0.5 

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images_gpu = list(image.to(device) for image in images)
            outputs = model(images_gpu)

            for i in range(len(images)):
                pred_boxes = outputs[i]['boxes'].cpu().numpy()
                pred_labels = outputs[i]['labels'].cpu().numpy()
                pred_scores = outputs[i]['scores'].cpu().numpy()
                
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()

                high_conf_idx = pred_scores > score_threshold
                p_boxes = pred_boxes[high_conf_idx]
                p_labels = pred_labels[high_conf_idx]

                matched_gt = []
                for pb, pl in zip(p_boxes, p_labels):
                    found_match = False
                    for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if idx in matched_gt: continue
                        
                        if pl == gl and calculate_iou(pb, gb) > iou_threshold:
                            correct_count += 1
                            matched_gt.append(idx)
                            found_match = True
                            break
                    
                    if not found_match:
                        wrong_count += 1

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS+1), history['valid_loss'], label='Valid Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    labels = ['Correct (TP)', 'Wrong (FP)']
    values = [correct_count, wrong_count]
    plt.bar(labels, values, color=['green', 'red'])
    plt.title(f'Detection Performance (Total: {correct_count + wrong_count})')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()