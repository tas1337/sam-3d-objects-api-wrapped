"""
SAM3D API Server - Simple: Upload image, get 3D GLB back
Auto-segments the image to find the object
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
rembg_session = None

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

def init_rembg():
    global rembg_session
    if rembg_session is not None:
        return rembg_session
    try:
        from rembg import new_session
        logger.info("Loading rembg...")
        rembg_session = new_session("u2net")
        logger.info("rembg loaded!")
        return rembg_session
    except ImportError:
        logger.warning("rembg not installed")
        return None

def process_image(data: dict) -> np.ndarray:
    """
    Load image and auto-segment if needed.
    Returns RGBA numpy array with proper alpha mask.
    """
    # Load image
    if 'image' in data:
        img_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(img_data))
    elif 'image_url' in data:
        import requests as req
        resp = req.get(data['image_url'], timeout=60)
        image = Image.open(io.BytesIO(resp.content))
    else:
        raise ValueError('Need image or image_url')
    
    # Check if already has alpha with actual transparency
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]
        if alpha.min() < 250:
            logger.info("Image has alpha channel, using as mask")
            return np.array(image)
    
    # Auto-segment with rembg
    session = init_rembg()
    if session:
        from rembg import remove
        logger.info("Auto-segmenting...")
        image = remove(image, session=session)
        logger.info("Segmentation done")
        return np.array(image)
    
    # Fallback: center mask
    logger.info("Using center mask (rembg not available)")
    image = image.convert('RGBA')
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 255
    img_array[:, :, 3] = mask
    return img_array

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
    """
    Upload image → Get 3D GLB file back
    
    Request: { "image": "base64...", "seed": 42 }
    or:      { "image_url": "https://...", "seed": 42 }
    
    Response: GLB file download
    """
    try:
        if inference is None:
            init_model()
        
        data = request.get_json()
        img_array = process_image(data)
        seed = data.get('seed', 42)
        
        mask = img_array[:, :, 3] > 127
        logger.info(f"Running 3D generation, seed={seed}, object pixels: {mask.sum()}")
        
        output = inference._pipeline.run(
            img_array, None, seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=True,
            use_vertex_color=False
        )
        
        glb = output.get("glb")
        if glb is None:
            return jsonify({'error': 'No mesh generated'}), 500
        
        tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
        glb.export(tmp.name)
        
        return send_file(
            tmp.name, 
            as_attachment=True, 
            download_name='model.glb', 
            mimetype='model/gltf-binary'
        )
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        init_model()
    except Exception as e:
        logger.error(f"Model init failed: {e}")
    
    try:
        init_rembg()
    except Exception as e:
        logger.warning(f"rembg init failed: {e}")
    
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
