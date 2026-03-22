import json, os, random, pandas as pd
from tqdm import tqdm
from config import Config

SENTIMENT_MAP = {"Positive": 0, "Neutral": 1, "Negative": 2}

def main():
    os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
    # 1. Load Triplets
    triplet_map = {}
    if os.path.exists(Config.TRIPLE_CSV):
        df_trips = pd.read_csv(Config.TRIPLE_CSV)
        for _, row in df_trips.iterrows():
            img_name = str(row['image']) + ".jpg" if ".jpg" not in str(row['image']) else str(row['image'])
            trip_text = f"{row['subject']} {row['relation']} {row['object']}"
            if img_name not in triplet_map: triplet_map[img_name] = []
            if len(triplet_map[img_name]) < 5:
                triplet_map[img_name].append({"text": trip_text, "sub": str(row['subject']).lower(), "obj": str(row['object']).lower()})

    # 2. Load Raw Data
    with open(Config.RAW_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    samples = []
    for entry in tqdm(raw_data, desc="Processing MASC"):
        img_filename = f"{entry.get('image_id')}.jpg"
        if not os.path.exists(os.path.join(Config.IMG_DIR, img_filename)): continue

        review_text = entry.get('review', "")
        caption = entry.get('photo_caption', "")
        triples = triplet_map.get(img_filename, [])
        aspects = entry.get('review_aspects', [])
        sentiments = entry.get('review_opinion_categories', [])

        for i, asp in enumerate(aspects):
            term = asp.get('term', '') if isinstance(asp, dict) else str(asp)
            sentiment_label = sentiments[i] if i < len(sentiments) else "Neutral"
            if term:
                samples.append({
                    "review_text": review_text, "aspect": term,
                    "label": SENTIMENT_MAP.get(sentiment_label, 1),
                    "image_id": img_filename, "caption": caption, "triples": triples
                })

    # 3. Shuffle & Save
    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    
    with open(Config.TRAIN_JSON, 'w') as f: json.dump(samples[:split], f, indent=4)
    with open(Config.TEST_JSON, 'w') as f: json.dump(samples[split:], f, indent=4)
    print(f"✅ Created MASC dataset: {len(samples)} samples. Files saved in {Config.PROCESSED_DIR}")

if __name__ == "__main__":
    main()
