"""
SAM3D API Client

Simple Python client for interacting with the SAM3D API.
Use this from your application to generate 3D meshes.

Example:
    from sam3d_client import SAM3DClient
    
    client = SAM3DClient("https://your-pod-id-8000.proxy.runpod.net")
    
    # Generate from file
    client.generate("image.png", "output.glb")
    
    # Generate from URL
    client.generate_from_url("https://example.com/image.jpg", "output.glb")
"""

import requests
import base64
from pathlib import Path
from typing import Optional, Literal
import time


class SAM3DClient:
    """Client for SAM3D API"""
    
    def __init__(self, base_url: str, timeout: int = 300):
        """
        Initialize client.
        
        Args:
            base_url: API base URL (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def health(self) -> dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def is_ready(self) -> bool:
        """Check if API is ready"""
        try:
            health = self.health()
            return health.get("model_loaded", False)
        except:
            return False
    
    def wait_until_ready(self, max_wait: int = 300, poll_interval: int = 5):
        """Wait until API is ready"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.is_ready():
                return True
            time.sleep(poll_interval)
        raise TimeoutError(f"API not ready after {max_wait}s")
    
    def generate(
        self,
        image_path: str,
        output_path: str = "output.glb",
        seed: int = 42,
        simplify: float = 0.95,
        texture_size: int = 1024,
        with_texture: bool = True,
        mask_mode: Literal["alpha", "center", "full"] = "alpha"
    ) -> dict:
        """
        Generate 3D mesh from image file.
        
        Args:
            image_path: Path to input image
            output_path: Path to save GLB output
            seed: Random seed
            simplify: Mesh simplification ratio
            texture_size: Texture resolution
            with_texture: Whether to bake textures
            mask_mode: Mask creation mode
            
        Returns:
            Response metadata (file_size_mb, inference_time_s, etc.)
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Make request
        payload = {
            "image": image_b64,
            "seed": seed,
            "simplify": simplify,
            "texture_size": texture_size,
            "with_texture": with_texture,
            "mask_mode": mask_mode
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            raise RuntimeError(f"Generation failed: {data.get('error')}")
        
        # Save GLB file
        model_data = base64.b64decode(data["model_data"])
        with open(output_path, "wb") as f:
            f.write(model_data)
        
        return {
            "output_path": output_path,
            "format": data.get("format"),
            "file_size_mb": data.get("file_size_mb"),
            "inference_time_s": data.get("inference_time_s")
        }
    
    def generate_from_url(
        self,
        image_url: str,
        output_path: str = "output.glb",
        seed: int = 42,
        simplify: float = 0.95,
        texture_size: int = 1024,
        with_texture: bool = True,
        mask_mode: Literal["alpha", "center", "full"] = "center"
    ) -> dict:
        """
        Generate 3D mesh from image URL.
        
        Args:
            image_url: URL of input image
            output_path: Path to save GLB output
            seed: Random seed
            simplify: Mesh simplification ratio
            texture_size: Texture resolution
            with_texture: Whether to bake textures
            mask_mode: Mask creation mode
            
        Returns:
            Response metadata
        """
        payload = {
            "image_url": image_url,
            "seed": seed,
            "simplify": simplify,
            "texture_size": texture_size,
            "with_texture": with_texture,
            "mask_mode": mask_mode
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            raise RuntimeError(f"Generation failed: {data.get('error')}")
        
        model_data = base64.b64decode(data["model_data"])
        with open(output_path, "wb") as f:
            f.write(model_data)
        
        return {
            "output_path": output_path,
            "format": data.get("format"),
            "file_size_mb": data.get("file_size_mb"),
            "inference_time_s": data.get("inference_time_s")
        }
    
    def generate_ply(
        self,
        image_path: str,
        output_path: str = "output.ply",
        seed: int = 42
    ) -> str:
        """
        Generate Gaussian Splat PLY from image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save PLY output
            seed: Random seed
            
        Returns:
            Output file path
        """
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        response = requests.post(
            f"{self.base_url}/generate-ply",
            json={"image": image_b64, "seed": seed},
            timeout=self.timeout
        )
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return output_path


# Convenience function for quick use
def generate_mesh(
    image_path: str,
    output_path: str = "output.glb",
    api_url: str = "http://localhost:8000",
    **kwargs
) -> dict:
    """
    Quick function to generate a 3D mesh.
    
    Args:
        image_path: Path to input image
        output_path: Path to save GLB output  
        api_url: API URL
        **kwargs: Additional options (seed, simplify, texture_size, etc.)
        
    Returns:
        Generation result metadata
    """
    client = SAM3DClient(api_url)
    return client.generate(image_path, output_path, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D mesh from image")
    parser.add_argument("image", help="Input image path or URL")
    parser.add_argument("-o", "--output", default="output.glb", help="Output GLB path")
    parser.add_argument("-u", "--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-texture", action="store_true", help="Skip texture baking")
    args = parser.parse_args()
    
    client = SAM3DClient(args.url)
    
    # Check if input is URL or file
    if args.image.startswith("http"):
        result = client.generate_from_url(
            args.image, args.output, 
            seed=args.seed, 
            with_texture=not args.no_texture
        )
    else:
        result = client.generate(
            args.image, args.output,
            seed=args.seed,
            with_texture=not args.no_texture
        )
    
    print(f"✅ Generated: {result['output_path']}")
    print(f"   Size: {result['file_size_mb']:.2f} MB")
    print(f"   Time: {result['inference_time_s']:.2f}s")

