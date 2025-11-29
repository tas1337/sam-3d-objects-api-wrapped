#!/usr/bin/env python3
"""
Test script for SAM3D API
Run this to verify the API is working correctly.
"""

import requests
import base64
import json
import sys
from pathlib import Path
import argparse


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_health(api_url: str) -> bool:
    """Test health endpoint"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print(f"   Model loaded: {data.get('model_loaded')}")
        print(f"   GPU: {data.get('gpu_name')}")
        return data.get('status') == 'healthy'
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_generate(api_url: str, image_path: str, output_path: str = "output.glb") -> bool:
    """Test /generate endpoint"""
    print(f"🎨 Testing /generate endpoint with {image_path}...")
    
    try:
        # Encode image
        image_b64 = encode_image(image_path)
        
        # Send request
        payload = {
            "image": image_b64,
            "seed": 42,
            "simplify": 0.95,
            "texture_size": 1024,
            "with_texture": True
        }
        
        print("   📤 Sending request...")
        response = requests.post(
            f"{api_url}/generate",
            json=payload,
            timeout=300  # 5 minutes for inference
        )
        
        if response.status_code != 200:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return False
        
        data = response.json()
        
        if not data.get('success'):
            print(f"   ❌ Error: {data.get('error')}")
            return False
        
        # Save GLB file
        model_data = base64.b64decode(data['model_data'])
        with open(output_path, 'wb') as f:
            f.write(model_data)
        
        print(f"   ✅ Success!")
        print(f"   📦 Saved to: {output_path}")
        print(f"   📊 Size: {data.get('file_size_mb', 0):.2f} MB")
        print(f"   ⏱️  Time: {data.get('inference_time_s', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_generate_file(api_url: str, image_path: str, output_path: str = "output_file.glb") -> bool:
    """Test /generate-file endpoint"""
    print(f"📥 Testing /generate-file endpoint with {image_path}...")
    
    try:
        image_b64 = encode_image(image_path)
        
        payload = {
            "image": image_b64,
            "seed": 42,
            "with_texture": True
        }
        
        response = requests.post(
            f"{api_url}/generate-file",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            print(f"   ❌ Error: {response.status_code}")
            return False
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   ✅ Success!")
        print(f"   📦 Saved to: {output_path}")
        print(f"   📊 Size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SAM3D API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--output", default="output.glb", help="Output GLB path")
    args = parser.parse_args()
    
    print(f"🔗 Testing API at: {args.url}")
    print("=" * 50)
    
    # Test health
    if not test_health(args.url):
        print("\n❌ Health check failed. Is the server running?")
        sys.exit(1)
    
    print("")
    
    # Test generation if image provided
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        
        if not test_generate(args.url, args.image, args.output):
            print("\n❌ Generation failed.")
            sys.exit(1)
    else:
        print("ℹ️  No image provided. Skipping generation test.")
        print("   Use --image <path> to test generation.")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()

