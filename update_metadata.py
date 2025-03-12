import json
import os
import re
from pathlib import Path

def update_metadata(metadata_path):
    # Load metadata
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    # Get the directory containing metadata.json
    base_dir = Path(metadata_path).parent

    updated_data = []
    for item in data:
        uuid = item['uuid']
        
        # Create new item with only required fields
        new_item = {
            'uuid': uuid,
            'audio_path': item['audio_path'],
            'lyrics_path': item['lyrics_path'],
            'lyrics': item['lyrics'],
            'description': item['description'],
            'clips': []
        }

        # Process each clip
        for clip in item['clips']:
            # Extract clip number from original path
            original_clip_path = clip['original_path']
            clip_match = re.search(r'clip_(\d+)\.mp3$', original_clip_path)
            if not clip_match:
                continue
            
            clip_num = clip_match.group(1)
            
            # Update paths to be relative to metadata.json location
            original_path = f"original/{uuid}/clip_{clip_num}.mp3"
            vocals_path = f"vocals/{uuid}/clip_{clip_num}.mp3"

            # Check if both files exist
            if os.path.exists(base_dir / original_path) and os.path.exists(base_dir / vocals_path):
                # Create new clip with updated paths and only required fields
                new_clip = {
                    'start_time_seconds': clip['start_time_seconds'],
                    'duration_seconds': clip['duration_seconds'],
                    'original_path': original_path,
                    'vocals_path': vocals_path,
                    'lyrics': clip['lyrics'],
                    'is_music': clip['is_music']
                }
                new_item['clips'].append(new_clip)

        # Only add items that have valid clips
        if new_item['clips']:
            updated_data.append(new_item)

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(updated_data, f, indent=2)

    print(f"Updated metadata saved to {metadata_path}")
    print(f"Total items processed: {len(data)}")
    print(f"Total items retained: {len(updated_data)}")

if __name__ == "__main__":
    metadata_path = "data/metadata.json"
    update_metadata(metadata_path) 