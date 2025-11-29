#!/usr/bin/env python3
"""
Local verification script - tests everything that doesn't need GPU
Run this before building Docker to catch errors early
"""

import sys
import os
from pathlib import Path

# Change to project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "notebook"))

def test_step(name, test_fn):
    """Run a test step and report result"""
    try:
        test_fn()
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("SAM3D Local Verification")
    print("=" * 60)
    all_passed = True
    
    # 1. Check checkpoints exist
    print("\n📁 Checking checkpoints...")
    def check_checkpoints():
        required = [
            "checkpoints/hf/pipeline.yaml",
            "checkpoints/hf/ss_generator.ckpt",
            "checkpoints/hf/slat_generator.ckpt",
            "checkpoints/hf/slat_decoder_mesh.ckpt",
        ]
        for f in required:
            if not Path(f).exists():
                raise FileNotFoundError(f"Missing: {f}")
    all_passed &= test_step("Checkpoints present", check_checkpoints)
    
    # 2. Check Python dependencies
    print("\n📦 Checking dependencies...")
    
    def check_flask():
        from flask import Flask
        from flask_cors import CORS
    all_passed &= test_step("Flask + CORS", check_flask)
    
    def check_numpy():
        import numpy as np
        from PIL import Image
    all_passed &= test_step("NumPy + PIL", check_numpy)
    
    def check_trimesh():
        import trimesh
    all_passed &= test_step("Trimesh (GLB export)", check_trimesh)
    
    def check_torch():
        import torch
        print(f"      PyTorch version: {torch.__version__}")
        print(f"      CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
    all_passed &= test_step("PyTorch", check_torch)
    
    # 3. Check SAM3D modules (import only, no model loading)
    print("\n🧠 Checking SAM3D modules...")
    
    def check_sam3d_core():
        import sam3d_objects
        from sam3d_objects.config import utils as config_utils
    all_passed &= test_step("sam3d_objects core", check_sam3d_core)
    
    def check_sam3d_pipeline():
        from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
        from sam3d_objects.pipeline.inference_utils import get_pose_decoder
    all_passed &= test_step("Inference pipeline", check_sam3d_pipeline)
    
    def check_sam3d_postprocess():
        from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
        # Check to_glb function exists
        assert hasattr(postprocessing_utils, 'to_glb')
    all_passed &= test_step("Postprocessing (GLB)", check_sam3d_postprocess)
    
    # 4. Check API server can be imported
    print("\n🌐 Checking API server...")
    
    def check_api():
        # Mock CUDA for import test
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["LIDRA_SKIP_INIT"] = "true"
        
        # Import API without starting server
        import importlib.util
        spec = importlib.util.spec_from_file_location("api_server", "api_server.py")
        api_module = importlib.util.module_from_spec(spec)
        # Don't execute, just check syntax
        import ast
        with open("api_server.py", "r") as f:
            ast.parse(f.read())
    all_passed &= test_step("API server syntax", check_api)
    
    # 5. Check Dockerfile
    print("\n🐳 Checking Docker files...")
    
    def check_dockerfile():
        assert Path("Dockerfile").exists()
        with open("Dockerfile", "r") as f:
            content = f.read()
            assert "nvidia/cuda" in content
            assert "sam3d-objects" in content
    all_passed &= test_step("Dockerfile", check_dockerfile)
    
    def check_compose():
        assert Path("docker-compose.yml").exists()
    all_passed &= test_step("docker-compose.yml", check_compose)
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. Build Docker image: ./scripts/build.sh")
        print("  2. Push to registry: ./scripts/push.sh")
        print("  3. Deploy on RunPod with GPU")
    else:
        print("❌ SOME CHECKS FAILED")
        print("Fix the issues above before building Docker")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

