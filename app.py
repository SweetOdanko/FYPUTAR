from flask import Flask, request, jsonify, render_template, send_file
from pymongo import MongoClient
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from models.model import SiameseNetwork
import io
import gridfs
from bson import ObjectId  
import os
import zipfile

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')


model = SiameseNetwork()
checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'))
model_state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
model.load_state_dict(model_state_dict, strict=False)
model.eval()

def get_db_connection(db_url):
    client = MongoClient(db_url)
    return client['image_search_engine']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results')
def results():
    image = request.args.get('image')
    db_url = request.args.get('db_url')
    return render_template('result.html', image=image, db_url=db_url)

@app.route('/api/upload', methods=['POST'])
def api_upload_image():
    db_url = request.form.get('db_url')
    if not db_url:
        return jsonify({'success': False, 'error': 'No database URL provided'}), 400
    
    db = get_db_connection(db_url)
    fs = gridfs.GridFS(db)
    collection = db['image_features']
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        if not file.mimetype.startswith('image/'):
            return jsonify({'success': False, 'error': 'Not supported image format!'}), 400

        try:
            image = Image.open(file.stream).convert('RGB')
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)

            file_id = fs.put(image_bytes, filename=file.filename, content_type='image/png')
            print(f"Image {file.filename} saved to GridFS with ID: {file_id}")
            
            image_url = f"/api/get_image/{file_id}?db_url={db_url}" 
            image = preprocess_image(image)
            features = extract_features(image)

            start_time = time.time()
            similar_images = find_similar_images(features, collection)
            processing_time = time.time() - start_time

            if not similar_images:
                return jsonify({'success': False, 'error': 'No similar images detected', 'image_url': image_url}), 200
            
            return jsonify({'success': True, 'images': similar_images, 'image_url': image_url, 'processing_time': processing_time})
        except Exception as e:
            print(f"Error during image processing and feature extraction: {str(e)}")
            return jsonify({'success': False, 'error': 'Error during image processing and feature extraction'}), 500

    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'success': False, 'error': 'General error occurred'}), 500

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  

def extract_features(image):
    with torch.no_grad():
        features = model.forward_once(image)
    return features.cpu().numpy()

def find_similar_images(features, collection):
    feature_vector = features.flatten().tolist()
    all_features = list(collection.find())
    similarities = []
    for item in all_features:
        db_features = np.array(item['feature_vector'])
        similarity = np.dot(feature_vector, db_features) / (np.linalg.norm(feature_vector) * np.linalg.norm(db_features))
        similarities.append((item['file_id'], similarity))
    
    similarity_threshold = 0.97

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    if similarities[0][1] >= similarity_threshold:
        similar_images = [str(similarities[i][0]) for i in range(6)]
    else:
        similar_images = [str(similarity[0]) for similarity in similarities if similarity[1] >= similarity_threshold]
        similar_images = similar_images[:6]  

    return similar_images


@app.route('/api/get_image/<image_id>', methods=['GET'])
def api_get_image(image_id):
    db_url = request.args.get('db_url')
    if not db_url:
        return "No database URL provided", 400
    
    db = get_db_connection(db_url)
    fs = gridfs.GridFS(db)
    
    try:
        print(f"Fetching image with ID: {image_id}")
        grid_out = fs.get(ObjectId(image_id))
        if grid_out is None:
            print(f"No file found in GridFS for ID: {image_id}")
            return "Image not found", 404

        print(f"Serving image with ID: {image_id}")
        mimetype = grid_out.content_type if grid_out.content_type else 'application/octet-stream'
        return send_file(io.BytesIO(grid_out.read()), mimetype=mimetype, as_attachment=False, download_name=grid_out.filename)
    except Exception as e:
        print(f"Error fetching image with ID {image_id}: {str(e)}")
        return str(e), 404

@app.route('/api/get_similar_images', methods=['GET'])
def api_get_similar_images():
    db_url = request.args.get('db_url')
    if not db_url:
        return jsonify({'success': False, 'error': 'No database URL provided'}), 400
    
    db = get_db_connection(db_url)
    fs = gridfs.GridFS(db)
    collection = db['image_features']
    
    image_url = request.args.get('image')
    if not image_url:
        return jsonify({'success': False, 'error': 'No image URL provided'}), 400
    
    image_path = image_url.split('?')[0]
    image_id = image_path.split('/')[-1]
    
    try:
        print(f"Fetching image with ID: {image_id}")
        grid_out = fs.get(ObjectId(image_id))
        image = Image.open(grid_out).convert('RGB')
        image = preprocess_image(image)
        features = extract_features(image)

        print(f"Extracted features shape: {features.shape}")
        print(f"Extracted features: {features}")

        similar_images = find_similar_images(features, collection)
        print(f"Similar images found: {similar_images}")

        return jsonify({'success': True, 'similar_images': similar_images})
    except Exception as e:
        print(f"Error occurred while fetching similar images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

import hashlib

@app.route('/api/upload_zip', methods=['POST'])
def upload_zip():
    db_url = request.form.get('db_url')
    if not db_url:
        return jsonify({'success': False, 'error': 'No database URL provided'}), 400

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if not (file.mimetype == 'application/zip'):
        return jsonify({'success': False, 'error': 'Not a zip file'}), 400

    db = get_db_connection(db_url)
    fs = gridfs.GridFS(db)
    collection = db['image_features']

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            results = []
            for zip_info in zip_ref.infolist():
                with zip_ref.open(zip_info) as image_file:
                    try:
                       
                        image_content = image_file.read()
                        image_hash = hashlib.md5(image_content).hexdigest()

                        
                        existing_file = collection.find_one({'image_hash': image_hash})
                        if existing_file:
                            print(f"Skipping {zip_info.filename} as a similar image already exists in the database")
                            continue

                        
                        image_file.seek(0)  
                        image = Image.open(image_file).convert('RGB')
                        image_tensor = preprocess_image(image)
                        features = extract_features(image_tensor)

                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='PNG')
                        image_bytes.seek(0)

                        timestamp = int(time.time())
                        new_image_name = f"{os.path.splitext(zip_info.filename)[0]}_{timestamp}.png"
                        file_id = fs.put(image_bytes, filename=new_image_name)
                        print(f"Image {new_image_name} saved to GridFS with ID: {file_id}")

                        feature_vector = features.flatten().tolist()
                        collection.insert_one({
                            'filename': new_image_name,
                            'feature_vector': feature_vector,
                            'file_id': file_id,
                            'image_hash': image_hash
                        })
                        print(f"Saved {new_image_name} to database with file_id {file_id}")

                        results.append({'file_id': str(file_id), 'filename': new_image_name})
                    except Exception as e:
                        print(f"Failed to process image {zip_info.filename}: {e}")

        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Failed to process zip file: {e}")
        return jsonify({'success': False, 'error': 'Failed to process zip file'}), 500

if __name__ == '__main__':
    app.run(debug=True)
