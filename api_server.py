"""
SAM3D API Server - GLB mesh output
"""
import os
import sys
import base64
import tempfile
from pathlib import Path

os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
os.environ["LIDRA_SKIP_INIT"] = "true"

sys.path.append("notebook")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

inference = None

def init_model():
    global inference
    if inference is not None:
        return inference
    
    logger.info("Loading SAM3D model...")
    from inference import Inference
    
    config_path = "checkpoints/hf/pipeline.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    inference = Inference(config_path, compile=False)
    logger.info("Model loaded!")
    return inference

@app.route('/health', methods=['GET'])
def health():
    import torch
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference is not None,
        'cuda': torch.cuda.is_available()
    })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if inference is None:
            init_model()
        
        data = request.get_json()
        
        # Get image
        if 'image' in data:
            img_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(img_data)).convert('RGBA')
        elif 'image_url' in data:
            import requests as req
            resp = req.get(data['image_url'], timeout=60)
            image = Image.open(io.BytesIO(resp.content)).convert('RGBA')
        else:
            return jsonify({'error': 'Need image or image_url'}), 400
        
        img_array = np.array(image)
        seed = data.get('seed', 42)
        
        # Create mask from alpha or center
        if img_array.shape[-1] == 4:
            mask = img_array[:, :, 3] > 127
        else:
            h, w = img_array.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = True
        
        # Run inference
        logger.info(f"Running inference, seed={seed}")
        output = inference._pipeline.run(
            img_array, None, seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=data.get('with_texture', True),
            with_layout_postprocess=True,
            use_vertex_color=not data.get('with_texture', True)
        )
        
        glb = output.get("glb")
        if glb is None:
            return jsonify({'error': 'No mesh generated'}), 500
        
        # Save GLB
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
            glb.export(f.name)
            with open(f.name, 'rb') as rf:
                glb_data = base64.b64encode(rf.read()).decode()
            os.unlink(f.name)
        
        return jsonify({
            'success': True,
            'model_data': glb_data,
            'format': 'glb'
        })
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/generate-file', methods=['POST'])
def generate_file():
    try:
        if inference is None:
            init_model()
        
        data = request.get_json()
        
        if 'image' in data:
            img_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(img_data)).convert('RGBA')
        elif 'image_url' in data:
            import requests as req
            resp = req.get(data['image_url'], timeout=60)
            image = Image.open(io.BytesIO(resp.content)).convert('RGBA')
        else:
            return jsonify({'error': 'Need image or image_url'}), 400
        
        img_array = np.array(image)
        seed = data.get('seed', 42)
        
        if img_array.shape[-1] == 4:
            mask = img_array[:, :, 3] > 127
        else:
            h, w = img_array.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = True
        
        output = inference._pipeline.run(
            img_array, None, seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=data.get('with_texture', True),
            with_layout_postprocess=True,
            use_vertex_color=not data.get('with_texture', True)
        )
        
        glb = output.get("glb")
        if glb is None:
            return jsonify({'error': 'No mesh generated'}), 500
        
        tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
        glb.export(tmp.name)
        
        return send_file(tmp.name, as_attachment=True, download_name='model.glb', mimetype='model/gltf-binary')
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        init_model()
    except Exception as e:
        logger.error(f"Model init failed: {e}")
    
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
