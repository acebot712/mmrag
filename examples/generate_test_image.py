#!/usr/bin/env python3
"""Generate a simple test image for MM-RAG demos."""

from PIL import Image, ImageDraw, ImageFont
import os

def generate_test_image(output_path: str = "examples/test_tower.png"):
    """Generate a simple image with text representing the Eiffel Tower."""
    # Create a blank image
    width, height = 512, 512
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)

    # Draw a simple tower shape
    # Base
    draw.rectangle([(200, 450), (312, 470)], fill='darkgray', outline='black', width=2)

    # First level (wide)
    points = [(180, 450), (256, 300), (332, 450)]
    draw.polygon(points, fill='gray', outline='black')

    # Second level (medium)
    points = [(220, 300), (256, 200), (292, 300)]
    draw.polygon(points, fill='darkgray', outline='black')

    # Third level (narrow)
    points = [(240, 200), (256, 100), (272, 200)]
    draw.polygon(points, fill='gray', outline='black')

    # Top spire
    draw.line([(256, 100), (256, 50)], fill='black', width=3)

    # Add text
    try:
        # Try to use default font, fall back if not available
        draw.text((150, 480), "Famous Landmark Tower", fill='black')
    except:
        pass

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    img.save(output_path)
    print(f"Test image saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_test_image()
