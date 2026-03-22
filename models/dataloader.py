import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class HotelExtractDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, image_dir):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_dir = image_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        entry = self.data[item]
        
        review = entry['review_text']   
        aspect_term = entry['aspect']  
        label = entry['label']         
        img_filename = entry['image_id']
        caption = entry['caption']    
        triples_data = entry['triples']

        # 1. TOKENIZE
        context_text = f"{caption}. {review}"
        ids_context = self.tokenizer.encode(context_text, add_special_tokens=False)
        
        triplets_ids = []
        for t in triples_data:
            t_ids = self.tokenizer.encode(t['text'], add_special_tokens=False)
            triplets_ids.append(t_ids)

        # 2. BUILD INPUT IDS
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        
        input_ids = [bos] + ids_context + [eos]
        range_context = (0, len(input_ids))
        
        range_triples = []
        for t_ids in triplets_ids:
            start = len(input_ids)
            input_ids.extend(t_ids + [eos])
            end = len(input_ids)
            range_triples.append((start, end))

        # Truncate
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            range_context = (0, min(range_context[1], self.max_len))
            range_triples = [(s, min(e, self.max_len)) for s, e in range_triples if s < self.max_len]

        padding_len = self.max_len - len(input_ids)
        real_len = len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        
        # 3. BUILD VISIBLE MATRIX
        visible_matrix = np.full((self.max_len, self.max_len), -1e9, dtype=np.float32)
        
        def set_visible(r1, r2):
            visible_matrix[r1[0]:r1[1], r2[0]:r2[1]] = 0.0

        set_visible(range_context, (0, self.max_len))
        set_visible((0, self.max_len), range_context)
        
        for r in range_triples:
            set_visible(r, r)
            
        for i in range(len(range_triples)):
            for j in range(i + 1, len(range_triples)):
                ents_i = {triples_data[i]['sub'], triples_data[i]['obj']}
                ents_j = {triples_data[j]['sub'], triples_data[j]['obj']}
                if not ents_i.isdisjoint(ents_j):
                    set_visible(range_triples[i], range_triples[j])
                    set_visible(range_triples[j], range_triples[i])
                    
        visible_matrix[real_len:, :] = -1e9
        visible_matrix[:, real_len:] = -1e9
        np.fill_diagonal(visible_matrix, 0.0)

        # 4. DECODER PROMPT
        decoder_text = f"The sentiment of {aspect_term} is <mask"
        dec = self.tokenizer.encode_plus(
            decoder_text, add_special_tokens=True, max_length=32,
            padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True
        )

        # 5. IMAGE LOADING
        try:
            image_path = os.path.join(self.image_dir, img_filename)
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(visible_matrix, dtype=torch.float),
            'decoder_input_ids': dec['input_ids'].flatten(),
            'decoder_attention_mask': dec['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
            'image_pixels': image
        }
