"""
RunPod Serverless Handler for SAM3D
This replaces the Flask API for serverless deployment
"""
import os
import sys
import base64
import tempfile
import time

# Set environment variables before imports
os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("notebook")

import runpod
import numpy as np
from PIL import Image
import io
import logging
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (persists between calls during warm state)
inference = None
rembg_session = None


def init_model():
    """Initialize the SAM3D model (called on cold start)"""
    global inference
    if inference is not None:
        return inference
    
    logger.info("Loading SAM3D model...")
    start = time.time()
    
    from inference import Inference
    
    possible_paths = [
        "checkpoints/pipeline.yaml",
        "checkpoints/hf/pipeline.yaml",
    ]
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        raise FileNotFoundError("No pipeline.yaml found in checkpoints/")
    
    logger.info(f"Using config: {config_path}")
    inference = Inference(config_path)
    
    logger.info(f"Model loaded in {time.time() - start:.1f}s")
    return inference


def init_rembg():
    """Initialize rembg for background removal"""
    global rembg_session
    if rembg_session is not None:
        return rembg_session
    try:
        from rembg import new_session
        rembg_session = new_session("u2net")
        logger.info("rembg initialized")
        return rembg_session
    except Exception as e:
        logger.warning(f"rembg not available: {e}")
        return None


def process_image(job_input):
    """Process input image from base64 or URL"""
    if 'image' in job_input:
        img_data = base64.b64decode(job_input['image'])
        img = Image.open(io.BytesIO(img_data))
    elif 'image_url' in job_input:
        import requests
        response = requests.get(job_input['image_url'], timeout=30)
        img = Image.open(io.BytesIO(response.content))
    else:
        raise ValueError("No image or image_url provided")
    
    # Convert to RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Remove background if rembg available
    session = init_rembg()
    if session:
        from rembg import remove
        img = remove(img, session=session)
        logger.info("Background removed with rembg")
    else:
        # Create center mask if no rembg
        logger.info("Using center mask (rembg not available)")
        arr = np.array(img)
        h, w = arr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 3
        y, x = np.ogrid[:h, :w]
        circle_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2
        mask[circle_mask] = 255
        arr[:, :, 3] = mask
        img = Image.fromarray(arr)
    
    return np.array(img)


def handler(job):
    """
    RunPod Serverless Handler
    
    Input:
    {
        "input": {
            "image": "base64_encoded_image",  # or "image_url": "https://..."
            "output_format": "glb",  # or "ply"
            "with_texture": true,
            "texture_size": 2048,
            "simplify": 0.3,
            "inference_steps": 50,
            "nviews": 200,
            "seed": 42
        }
    }
    
    Output:
    {
        "model": "base64_encoded_glb",
        "format": "glb",
        "vertices": 12345,
        "faces": 24690,
        "processing_time": 45.2
    }
    """
    job_input = job["input"]
    start_time = time.time()
    
    try:
        # Initialize model (cold start or cached)
        model = init_model()
        
        # Process input image
        img_array = process_image(job_input)
        
        # Get parameters
        seed = job_input.get('seed', 42)
        output_format = job_input.get('output_format', 'glb')
        with_texture = job_input.get('with_texture', True)
        
        # Quality parameters
        if with_texture:
            texture_size = job_input.get('texture_size', 2048)
            simplify = job_input.get('simplify', 0.3)
            inference_steps = job_input.get('inference_steps', 50)
            nviews = job_input.get('nviews', 200)
        else:
            texture_size = job_input.get('texture_size', 2048)
            simplify = job_input.get('simplify', 0.0)
            inference_steps = job_input.get('inference_steps', 50)
            nviews = job_input.get('nviews', 200)
        
        mask = img_array[:, :, 3] > 127
        logger.info(f"Running 3D generation, seed={seed}, pixels={mask.sum()}, quality: texture_size={texture_size}, simplify={simplify}, inference_steps={inference_steps}, nviews={nviews}")
        
        # Clear GPU memory before generation
        gc.collect()
        torch.cuda.empty_cache()
        
        if output_format == 'ply':
            # Generate Gaussian splat
            output = model._pipeline.run(
                img_array, None, seed,
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=True,
                use_vertex_color=False
            )
            
            gs = output.get("gs")
            if gs is None:
                raise ValueError("No gaussian splat generated")
            
            tmp = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
            tmp.close()
            gs.save_ply(tmp.name)
            
            with open(tmp.name, 'rb') as f:
                result_data = base64.b64encode(f.read()).decode('utf-8')
            
            os.unlink(tmp.name)
            
            return {
                "model": result_data,
                "format": "ply",
                "processing_time": round(time.time() - start_time, 1)
            }
        
        else:
            # Generate GLB mesh
            stage1_steps = inference_steps
            stage2_steps = inference_steps
            
            # Skip pipeline's postprocessing, do our own
            output = model._pipeline.run(
                img_array, None, seed,
                stage1_only=False,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=True,
                use_vertex_color=False,
                stage1_inference_steps=stage1_steps,
                stage2_inference_steps=stage2_steps
            )
            
            if "mesh" not in output or "gaussian" not in output:
                raise ValueError("Pipeline did not produce mesh or gaussian output")
            
            from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
            
            # Clear GPU memory before texture baking
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set render quality parameters
            do_mesh_postprocess = with_texture
            postprocessing_utils.to_glb._render_resolution = min(texture_size, 2048)
            postprocessing_utils.to_glb._render_nviews = nviews
            
            glb = postprocessing_utils.to_glb(
                output["gaussian"][0],
                output["mesh"][0],
                simplify=simplify if do_mesh_postprocess else 0.0,
                texture_size=texture_size,
                verbose=True,
                with_mesh_postprocess=do_mesh_postprocess,
                with_texture_baking=with_texture,
                use_vertex_color=not with_texture,
                rendering_engine=model._pipeline.rendering_engine,
            )
            
            if glb is None:
                raise ValueError("No mesh generated")
            
            # Export to temp file
            tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            tmp.close()
            glb.export(tmp.name)
            
            # Get mesh stats
            vertices = len(glb.vertices)
            faces = len(glb.faces)
            
            # Read and encode
            with open(tmp.name, 'rb') as f:
                result_data = base64.b64encode(f.read()).decode('utf-8')
            
            os.unlink(tmp.name)
            
            # Cleanup
            del glb, output
            gc.collect()
            torch.cuda.empty_cache()
            
            processing_time = round(time.time() - start_time, 1)
            logger.info(f"Completed in {processing_time}s, vertices={vertices}, faces={faces}")
            
            return {
                "model": result_data,
                "format": "glb",
                "vertices": vertices,
                "faces": faces,
                "processing_time": processing_time
            }
    
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        
        # Cleanup on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"error": str(e)}


# Start the serverless handler
runpod.serverless.start({"handler": handler})

