"""
SAM3D API Server - Queue-based processing
Only processes 1 request at a time, others wait in queue
"""
import os
import sys
import base64
import tempfile
import uuid
import threading
import time
from pathlib import Path
from queue import Queue
from collections import OrderedDict

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

# ============================================================================
# QUEUE SYSTEM
# ============================================================================
MAX_CONCURRENT = int(os.environ.get('MAX_CONCURRENT', 1))  # Process 1 at a time
MAX_QUEUE_SIZE = int(os.environ.get('MAX_QUEUE_SIZE', 10))  # Max waiting jobs

job_queue = Queue(maxsize=MAX_QUEUE_SIZE)
jobs = OrderedDict()  # job_id -> job info
jobs_lock = threading.Lock()
current_job = None

class Job:
    def __init__(self, job_id, data):
        self.id = job_id
        self.data = data
        self.status = 'queued'  # queued, processing, completed, failed
        self.position = 0
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None

def get_queue_position(job_id):
    """Get position in queue (0 = processing, 1 = next, etc.)"""
    with jobs_lock:
        if job_id not in jobs:
            return -1
        position = 0
        for jid, job in jobs.items():
            if job.status in ('queued', 'processing'):
                if jid == job_id:
                    return position
                position += 1
        return -1

def get_queue_stats():
    """Get queue statistics"""
    with jobs_lock:
        queued = sum(1 for j in jobs.values() if j.status == 'queued')
        processing = sum(1 for j in jobs.values() if j.status == 'processing')
        return {
            'queued': queued,
            'processing': processing,
            'max_queue_size': MAX_QUEUE_SIZE,
            'max_concurrent': MAX_CONCURRENT
        }

# ============================================================================
# MODEL & PROCESSING
# ============================================================================
inference = None
rembg_session = None

def init_model():
    global inference
    if inference is not None:
        return inference
    
    logger.info("Loading SAM3D model...")
    from inference import Inference
    
    possible_paths = [
        "checkpoints/pipeline.yaml",
        "checkpoints/hf/pipeline.yaml",
    ]
    config_path = None
    for path in possible_paths:
        if Path(path).exists():
            config_path = path
            break
    
    if not config_path:
        raise FileNotFoundError(f"Config not found! Tried: {possible_paths}")
    
    logger.info(f"Using config: {config_path}")
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
    """Load image and auto-segment if needed."""
    if 'image' in data:
        img_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(img_data))
    elif 'image_url' in data:
        import requests as req
        resp = req.get(data['image_url'], timeout=60)
        image = Image.open(io.BytesIO(resp.content))
    else:
        raise ValueError('Need image or image_url')
    
    if image.mode == 'RGBA':
        alpha = np.array(image)[:, :, 3]
        if alpha.min() < 250:
            logger.info("Image has alpha channel, using as mask")
            return np.array(image)
    
    session = init_rembg()
    if session:
        from rembg import remove
        logger.info("Auto-segmenting...")
        image = remove(image, session=session)
        logger.info("Segmentation done")
        return np.array(image)
    
    logger.info("Using center mask (rembg not available)")
    image = image.convert('RGBA')
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 255
    img_array[:, :, 3] = mask
    return img_array

def run_generation(job: Job):
    """Run the actual 3D generation"""
    global current_job
    
    try:
        if inference is None:
            init_model()
        
        data = job.data
        img_array = process_image(data)
        seed = data.get('seed', 42)
        output_format = data.get('output_format', 'glb')
        with_texture = data.get('with_texture', True)
        
        # Quality parameters (defaults set to HIGH quality - stable)
        texture_size = data.get('texture_size', 2048)  # Higher = better texture quality (1024, 2048, 4096). Default: 2048 (high quality, stable)
        simplify = data.get('simplify', 0.0)  # Lower = more mesh detail (0.0 = no simplification/max detail, 0.95 = aggressive). Default: 0.0 (max detail)
        inference_steps = data.get('inference_steps', 50)  # More steps = better quality (25 = fast, 50 = high, 100 = ultra). Default: 50 (high quality, stable)
        nviews = data.get('nviews', 200)  # More views = better texture (100 = default, 200 = high, 300 = ultra). Default: 200 (high quality, stable)
        remove_invisible_faces = data.get('remove_invisible_faces', True)  # Remove faces not visible from any angle. False = keep all faces (more detail but larger file)
        fill_holes_resolution = data.get('fill_holes_resolution', 2048)  # Higher = better hole detection (1024, 2048, 4096). Default: 2048
        fill_holes_num_views = data.get('fill_holes_num_views', 2000)  # More views = better hole detection (1000, 2000, 3000). Default: 2000
        
        mask = img_array[:, :, 3] > 127
        logger.info(f"[{job.id}] Running 3D generation, seed={seed}, pixels={mask.sum()}, quality: texture_size={texture_size}, simplify={simplify}, inference_steps={inference_steps}, nviews={nviews}")
        
        if output_format == 'ply':
            output = inference._pipeline.run(
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
            tmp.close()  # Close file handle before saving
            gs.save_ply(tmp.name)
            job.result = {'file': tmp.name, 'format': 'ply'}
        else:
            # Split inference_steps for stage1 and stage2
            stage1_steps = inference_steps
            stage2_steps = inference_steps
            
            # IMPORTANT: Set with_mesh_postprocess=False and with_texture_baking=False
            # to skip the pipeline's default postprocessing. We do our own with custom quality params.
            output = inference._pipeline.run(
                img_array, None, seed,
                stage1_only=False,
                with_mesh_postprocess=False,  # Skip - we do our own
                with_texture_baking=False,     # Skip - we do our own
                with_layout_postprocess=True,
                use_vertex_color=False,
                stage1_inference_steps=stage1_steps,
                stage2_inference_steps=stage2_steps
            )
            
            # Do postprocessing with our custom quality parameters
            if "mesh" not in output or "gaussian" not in output:
                raise ValueError("Pipeline did not produce mesh or gaussian output")
            
            from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
            
            logger.info(f"[{job.id}] Quality: texture_size={texture_size}, nviews={nviews}, simplify={simplify}, inference_steps={inference_steps}")
            
            # Set render quality parameters
            postprocessing_utils.to_glb._render_resolution = min(texture_size, 2048)
            postprocessing_utils.to_glb._render_nviews = nviews
            
            glb = postprocessing_utils.to_glb(
                output["gaussian"][0],
                output["mesh"][0],
                simplify=simplify,
                texture_size=texture_size,
                verbose=True,
                with_mesh_postprocess=True,
                with_texture_baking=with_texture,
                use_vertex_color=not with_texture,
                rendering_engine=inference._pipeline.rendering_engine,
            )
            
            if glb is None:
                raise ValueError("No mesh generated")
            
            # Export GLB file (this can be memory intensive)
            tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            tmp.close()  # Close file handle before exporting
            try:
                logger.info(f"[{job.id}] Exporting GLB file...")
                glb.export(tmp.name)
                logger.info(f"[{job.id}] GLB export completed: {tmp.name}")
            except Exception as export_error:
                logger.error(f"[{job.id}] GLB export failed: {export_error}", exc_info=True)
                # Try to clean up
                try:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
                except:
                    pass
                raise export_error
            
            job.result = {'file': tmp.name, 'format': 'glb'}
        
        # Mark as completed BEFORE cleanup to ensure status is saved
        job.status = 'completed'
        job.completed_at = time.time()
        logger.info(f"[{job.id}] Completed in {job.completed_at - job.started_at:.1f}s")
        
        # Clear large objects before garbage collection
        try:
            if 'glb' in locals():
                del glb
            if 'output' in locals():
                del output
            if 'img_array' in locals():
                del img_array
        except:
            pass
        
        # Force garbage collection and CUDA cache clear
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"[{job.id}] Cleanup completed, result ready at {job.result.get('file', 'N/A')}")
        
    except Exception as e:
        job.status = 'failed'
        job.error = str(e)
        job.completed_at = time.time()
        logger.error(f"[{job.id}] Failed: {e}", exc_info=True)
        
        # Force garbage collection and CUDA cleanup even on error
        import gc
        import torch
        try:
            if 'glb' in locals():
                del glb
            if 'output' in locals():
                del output
            if 'img_array' in locals():
                del img_array
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

def worker():
    """Worker thread that processes jobs one at a time"""
    global current_job
    logger.info("Queue worker started")
    
    while True:
        try:
            job_id = job_queue.get()
            
            with jobs_lock:
                if job_id not in jobs:
                    job_queue.task_done()
                    continue
                job = jobs[job_id]
                job.status = 'processing'
                job.started_at = time.time()
                current_job = job_id
            
            logger.info(f"[{job_id}] Processing started")
            run_generation(job)
            
            # Clear current_job before task_done to prevent deadlock
            with jobs_lock:
                current_job = None
            
            # Mark task as done (this can block if queue is full)
            job_queue.task_done()
            
            # Log completion for debugging
            logger.info(f"[{job_id}] Worker finished processing, waiting for next job")
        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].status = 'failed'
                    jobs[job_id].error = f"Worker error: {str(e)}"
                current_job = None
            try:
                job_queue.task_done()
            except:
                pass

# Start worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

def check_worker_thread():
    """Check if worker thread is alive, restart if dead"""
    global worker_thread
    while True:
        time.sleep(30)  # Check every 30 seconds
        if not worker_thread.is_alive():
            logger.error("Worker thread died! Restarting...")
            worker_thread = threading.Thread(target=worker, daemon=True)
            worker_thread.start()
            logger.info("Worker thread restarted")

# Start worker monitor thread
worker_monitor_thread = threading.Thread(target=check_worker_thread, daemon=True)
worker_monitor_thread.start()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - must be fast and non-blocking"""
    try:
        import torch
        gpu_info = {}
        if torch.cuda.is_available():
            try:
                gpu_info = {
                    'name': torch.cuda.get_device_name(0),
                    'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                }
            except:
                gpu_info = {'name': 'Unknown', 'vram_gb': 0}
        
        worker_alive = False
        if 'worker_thread' in globals():
            try:
                worker_alive = worker_thread.is_alive()
            except:
                pass
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': inference is not None,
            'cuda': torch.cuda.is_available(),
            'gpu': gpu_info,
            'queue': get_queue_stats(),
            'worker_alive': worker_alive
        })
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/queue', methods=['GET'])
def queue_status():
    """Get queue status - lightweight, doesn't trigger worker restart"""
    # This endpoint is intentionally lightweight to avoid triggering
    # Gunicorn's max-requests restart during job processing
    return jsonify(get_queue_stats())

@app.route('/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """Check status of a specific job"""
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        position = get_queue_position(job_id)
        
        response = {
            'job_id': job_id,
            'status': job.status,
            'position': position,
            'queue_length': get_queue_stats()['queued']
        }
        
        if job.status == 'completed':
            response['download_url'] = f'/job/{job_id}/download'
            response['processing_time'] = round(job.completed_at - job.started_at, 1)
        elif job.status == 'failed':
            response['error'] = job.error
        elif job.status == 'processing':
            response['message'] = 'Your job is being processed now'
        elif job.status == 'queued':
            response['message'] = f'Waiting in queue, position {position}'
        
        return jsonify(response)

@app.route('/job/<job_id>/download', methods=['GET'])
def job_download(job_id):
    """Download completed job result"""
    try:
        with jobs_lock:
            if job_id not in jobs:
                return jsonify({'error': 'Job not found'}), 404
            
            job = jobs[job_id]
            
            if job.status != 'completed':
                return jsonify({'error': f'Job not ready, status: {job.status}'}), 400
            
            if not job.result or 'file' not in job.result:
                return jsonify({'error': 'Job result file not found'}), 404
            
            result = job.result
            file_path = result['file']
        
        # Verify file exists before sending
        if not os.path.exists(file_path):
            logger.error(f"[{job_id}] File not found: {file_path}")
            return jsonify({'error': 'Result file no longer exists'}), 404
        
        if result['format'] == 'ply':
            return send_file(
                file_path,
                as_attachment=True,
                download_name='model.ply',
                mimetype='application/octet-stream'
            )
        else:
            return send_file(
                file_path,
                as_attachment=True,
                download_name='model.glb',
                mimetype='model/gltf-binary'
            )
    except Exception as e:
        logger.error(f"[{job_id}] Download failed: {e}", exc_info=True)
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """
    Submit a job to the queue
    
    Request: { "image_url": "https://...", "output_format": "glb", "with_texture": true }
    
    Response: { "job_id": "abc123", "position": 3, "status_url": "/job/abc123" }
    """
    data = request.get_json()
    
    # Validate input
    if 'image' not in data and 'image_url' not in data:
        return jsonify({'error': 'Need image or image_url'}), 400
    
    # Check queue capacity
    stats = get_queue_stats()
    if stats['queued'] >= MAX_QUEUE_SIZE:
        return jsonify({
            'error': 'Queue full, try again later',
            'queue_length': stats['queued'],
            'max_queue_size': MAX_QUEUE_SIZE
        }), 503
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, data)
    
    with jobs_lock:
        jobs[job_id] = job
    
    job_queue.put(job_id)
    position = get_queue_position(job_id)
    
    logger.info(f"[{job_id}] Job queued at position {position}")
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'position': position,
        'status_url': f'/job/{job_id}',
        'message': f'Job queued at position {position}. Poll status_url for updates.'
    })

@app.route('/generate/sync', methods=['POST'])
def generate_sync():
    """
    Synchronous generation - waits for result (may timeout for long queues)
    Same as old /generate behavior
    """
    data = request.get_json()
    
    if 'image' not in data and 'image_url' not in data:
        return jsonify({'error': 'Need image or image_url'}), 400
    
    # Create and queue job
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, data)
    
    with jobs_lock:
        jobs[job_id] = job
    
    job_queue.put(job_id)
    logger.info(f"[{job_id}] Sync job queued")
    
    # Wait for completion (with timeout)
    timeout = 600  # 10 minutes
    start = time.time()
    while time.time() - start < timeout:
        with jobs_lock:
            if job.status == 'completed':
                result = job.result
                break
            elif job.status == 'failed':
                return jsonify({'error': job.error}), 500
        time.sleep(1)
    else:
        return jsonify({'error': 'Timeout waiting for job'}), 504
    
    # Return file
    if result['format'] == 'ply':
        return send_file(
            result['file'],
            as_attachment=True,
            download_name='model.ply',
            mimetype='application/octet-stream'
        )
    else:
        return send_file(
            result['file'],
            as_attachment=True,
            download_name='model.glb',
            mimetype='model/gltf-binary'
        )

# Cleanup old jobs periodically
def cleanup_old_jobs():
    """Remove jobs older than 1 hour"""
    while True:
        time.sleep(300)  # Every 5 minutes
        cutoff = time.time() - 3600  # 1 hour
        with jobs_lock:
            to_remove = [jid for jid, job in jobs.items() 
                        if job.completed_at and job.completed_at < cutoff]
            for jid in to_remove:
                job = jobs.pop(jid)
                # Clean up temp file
                if job.result and 'file' in job.result:
                    try:
                        os.unlink(job.result['file'])
                    except:
                        pass
                logger.info(f"[{jid}] Cleaned up old job")

cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleanup_thread.start()

# ============================================================================
# MAIN
# ============================================================================

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
    logger.info(f"Queue settings: max_concurrent={MAX_CONCURRENT}, max_queue_size={MAX_QUEUE_SIZE}")
    app.run(host='0.0.0.0', port=port, threaded=True)
