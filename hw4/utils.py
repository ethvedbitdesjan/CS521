from skimage.segmentation import slic
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_interpretable_image(image: Image.Image, compactness: int=15, n_segments: int=10) -> np.array:
    image_np = np.array(image)
    segments = slic(image_np, n_segments=n_segments, compactness=compactness)
    
    num_segments = len(np.unique(segments))
    x_prime = np.ones(num_segments, dtype=bool) # Initialize x' to all 1s of bianry vector of each feature
    
    return x_prime, segments

def generate_samples(x_prime: np.array, segments: np.array, prob_feature: float=0.75, num_samples: int=100):
    num_segments = len(np.unique(segments))
    assert len(x_prime) == num_segments, "Length of x_prime should be equal to number of segments"
    
    samples = np.random.binomial(1, prob_feature, (num_samples, num_segments))
    
    return samples

def generate_sample_images(image: Image.Image, segments: np.array, x_prime: np.array, samples: np.array) -> np.array:
    assert samples.ndim == 2, "Samples should be a 2D array"
    num_segments = len(np.unique(segments))
    assert len(x_prime) == num_segments, "Length of x_prime should be equal to number of segments"
    assert samples.shape[1] == num_segments, "Number of segments in samples should be equal to number of segments"
    
    image_np = np.array(image)
    
    sample_images = []
    for sample in samples:
        sample_image = np.copy(image_np)
        for segment_idx, present in enumerate(sample):
            if not present:
                sample_image[segments == segment_idx] = 0 #turn off
        sample_images.append(sample_image)
    sample_images = np.stack(sample_images)
    return sample_images

def display_samples(samples, orig_img):
    assert len(samples) < 6, "Number of samples should be less than 6"
    fig, axs = plt.subplots(1, len(samples)+1, figsize=(20, 10))
    axs[0].imshow(orig_img)
    axs[0].axis('off')
    for i, sample in enumerate(samples):
        axs[i+1].imshow(sample)
        axs[i+1].axis('off')
    plt.show()

def image_similarity_kernel(image1, image2, sigma=1):
    image1 = image1.astype(float) / 255.0
    image2 = image2.astype(float) / 255.0
    l2_norm = np.linalg.norm((image1 - image2).flatten(), ord=2)
    l2_norm = l2_norm / np.sqrt(image1.flatten().shape[0])
    return np.exp(-((l2_norm/sigma)**2))

def visualize_explanations(image, explanations: list, titles: list, suptitle=None, overlay=True):
    assert len(explanations) == len(titles), "Number of explanations should be equal to number of titles"
    assert len(explanations) > 0, "At least one explanation should be provided"
    
    global_min = min([explanation.min() for explanation in explanations])
    global_max = max([explanation.max() for explanation in explanations])
    
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, len(explanations) + 2, width_ratios=[1] * (len(explanations) + 1) + [0.05])
    axes = []
    for i in range(len(explanations) + 1):
        axes.append(fig.add_subplot(gs[0, i]))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    for i, (explanation, title) in enumerate(zip(explanations, titles)):
        if overlay:
            axes[i+1].imshow(image)
        heatmap = axes[i+1].imshow(explanation, cmap='viridis', vmin=global_min, vmax=global_max)
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
    
    cbar_ax = fig.add_subplot(gs[0, -1])
    plt.colorbar(heatmap, cax=cbar_ax)

    if title is not None:
        plt.suptitle(f"{suptitle}")
    
    plt.tight_layout()
    plt.show()