# Download images for mangrove and non-mangrove classes using icrawler
from icrawler.builtin import GoogleImageCrawler
import os

# Define your dataset structure
categories = {
    'mangrove': [
        'mangrove forest',
        'mangrove trees',
        'mangrove ecosystem',
        'mangrove aerial view'
    ],
    'forest': [
        'forest trees',
        'dense forest',
        'tropical forest',
        'temperate forest',
        'coniferous forest',
        'broadleaf forest',
        'deciduous forest',
        'forest canopy',
        'forest landscape',
        'forest aerial view'
    ]
}

os.makedirs('dataset', exist_ok=True)

for label, keywords in categories.items():
    save_dir = os.path.join('dataset', label)
    os.makedirs(save_dir, exist_ok=True)
    crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    total_downloaded = 0
    for keyword in keywords:
        if total_downloaded >= 100:
            break
        print(f"Downloading {keyword} into {save_dir}")
        # Download up to the remaining needed images for this class
        remaining = 100 - total_downloaded
        crawler.crawl(keyword=keyword, max_num=remaining)
        # Count new images
        total_downloaded = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
