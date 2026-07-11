import os
import csv

def create_csv(base, labels, output_csv):
    rows = []
    for label in labels:
        folder = os.path.join(base, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder, file)
                rows.append([img_path, label])
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        writer.writerows(rows)
    print(f"CSV created: {output_csv} (Total images: {len(rows)})")

base = 'valid'
all_labels = os.listdir(base)
potato_labels = [d for d in all_labels if d.lower().startswith('potato')]
tomato_labels = [d for d in all_labels if d.lower().startswith('tomato')]

create_csv(base, potato_labels, 'potato_leaf_dataset.csv')
create_csv(base, tomato_labels, 'tomato_leaf_dataset.csv') 