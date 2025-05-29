import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

from common.config import IDPA_DIR

from typing import List

def crop(img, cropx=256, cropy=256, percent_x=0.5, percent_y=0.5):
        """
        Crop the image to (cropx, cropy) centered at (percent_x, percent_y) of the image dimensions.
        percent_x and percent_y should be between 0.0 and 1.0.
        """
        y, x = img.shape[:2]
        centerx = int(x * percent_x)
        centery = int(y * percent_y)
        startx = max(centerx - cropx // 2, 0)
        starty = max(centery - cropy // 2, 0)
        endx = min(startx + cropx, x)
        endy = min(starty + cropy, y)
        return img[starty:endy, startx:endx]

def get_original(key: str, split: str = "test"):
    return IDPA_DIR / split / "original" / "00000" / f"{key}.png"

def get_blurred(key: str, blur_type: str, split: str = "test"):
    return IDPA_DIR / split / "blurred" / blur_type / f"{key}.png"

def get_deblurred(key: str, method: str, blur_type: str, split: str = "test"):
    deblurred_dir = IDPA_DIR / split / "deblurred"
    matches = []
    for path in deblurred_dir.glob(f"{method}*"):
        if path.is_dir():
            matches.append(path / blur_type / f"{key}.png")

    return matches

def show_comparison(key: str, method: str, blur_types: List[str], cropped=False):
    for blur_type in blur_types:
        original = mpimg.imread(get_original(key))
        blurred = mpimg.imread(get_blurred(key, blur_type))
        deblurred_paths = get_deblurred(key, method, blur_type)
        deblurred = [mpimg.imread(path) for path in deblurred_paths]

        if cropped:
            original = crop(original)
            blurred = crop(blurred)
            deblurred = [crop(el) for el in deblurred]

        deblurred_methods = []
        for path in deblurred_paths:
            match = re.search(r'/([^/]*' + re.escape(method) + r'[^/]*)/', str(path))
            if match:
                deblurred_methods.append(match.group(1))

        images = [
            {"title": "Original", "image": original},
            {"title": f"Blurred ({blur_type})", "image": blurred}
        ]

        for i, img in enumerate(deblurred):
            images.append({
                "title": f"Deblurred {deblurred_methods[i]}" if len(deblurred) > 1 else f"Deblurred {method}",
                "image": img
            })

        _, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        if len(images) == 1:
            axes = [axes]
        for i, el in enumerate(images):
            axes[i].imshow(el["image"])
            axes[i].set_title(el["title"])
            axes[i].axis('off')
        plt.show()

def show_image(key: str, blur_type: str = None, method: str = None):
    path = None
    name = f"Image {key}"

    if blur_type:
        if method:
            path = get_deblurred(key, method, blur_type)[0]
            name += f" ({method}, {blur_type})"
        else:
            path = get_blurred(key, blur_type)
            name += f" ({blur_type})"
    else:
        path = get_original(key)

    image = mpimg.imread(path)

    plt.imshow(image)
    plt.title(f"Image {key}")
    plt.axis('off')
    plt.show()