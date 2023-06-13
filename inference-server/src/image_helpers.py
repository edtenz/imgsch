import os

from PIL import Image


def resize_image(image_path: str, width: int, height: int, output_dir: str, quality=70) -> str:
    """
    Resize the image to width and height and save it to the output_dir
    Args:
        image_path: image path
        width: new width
        height: new height
        output_dir: output directory
        quality: quality of the resized image, 1-100

    Returns: resized image path

    """
    image = Image.open(image_path)
    if width == 0 or height == 0:
        width, height = image.size

    resized_image = image.resize((width, height))

    # Create the 'resize' directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original image filename
    image_filename = os.path.basename(image_path)

    # Generate the output image path in the 'resize' directory
    output_path = os.path.join(output_dir, image_filename)

    resized_image.save(output_path, optimize=True, quality=quality)
    return output_path


def thumbnail(image_path: str, max_size: int, output_dir: str, quality=70) -> str:
    """
    Thumbnail the image to max_size and save it to the output_dir
    Args:
        image_path: image path
        max_size: max size of the image after resizing, width or height
        output_dir: path to the output directory
        quality: quality of the resized image, 1-100
    Returns: thumbnail image path
    """
    image = Image.open(image_path)
    image.thumbnail((max_size, max_size))

    # Create the 'resize' directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original image filename
    image_filename = os.path.basename(image_path)

    # Generate the output image path in the 'resize' directory
    output_path = os.path.join(output_dir, image_filename)

    image.save(output_path, optimize=True, quality=quality)
    return output_path
