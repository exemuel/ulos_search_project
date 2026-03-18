import os
import pickle
from PIL import Image
from core.extractor import FeatureExtractor

def build_index(image_folder, output_file="embeddings.pkl"):
    print("Initializing AI Model...")
    extractor = FeatureExtractor()
    
    features_db = []
    
    # Iterate through all images in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            try:
                img = Image.open(img_path)
                vector = extractor.extract(img)
                
                # Store the vector and the path to the image
                features_db.append({
                    "path": img_path,
                    "vector": vector
                })
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save the database to a file
    with open(output_file, 'wb') as f:
        pickle.dump(features_db, f)
    
    print(f"Indexing complete! Saved {len(features_db)} vectors to {output_file}")

if __name__ == "__main__":
    # Make sure you have images in this folder before running!
    build_index("data/ulos_images/")