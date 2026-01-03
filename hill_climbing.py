"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
import os


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """

    # Make prediction
    preds = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    decoded = decode_predictions(preds)[0]

    # Find probability of target label
    target_prob = 0.0
    for _, class_name, prob in decoded:
        if class_name == target_label:
            target_prob = prob
            break

    # Retrieve top prediction
    _, top_pred_label, top_prob = decoded[0]

    # If target_label is the top prediction return its probability
    # otherwise return negative of top prediction probability
    if top_pred_label == target_label:
        return target_prob
    else:
        return -top_prob


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Pixel values must remain in [0, 255]

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """

    neighbors = []
    height, width, channels = seed.shape
    max_pixel_change = 255 * epsilon

    # Strategy 1: Single pixel mutation (generate 5 variants)
    for _ in range(5):
        neighbor = seed.copy()
        # Randomly select a pixel
        i = random.randint(0, height - 1)
        j = random.randint(0, width - 1)
        c = random.randint(0, channels - 1)

        perturbation = random.uniform(-max_pixel_change, max_pixel_change)
        neighbor[i, j, c] += perturbation
        neighbors.append(neighbor)

    # Strategy 2: 3x3 patch mutation (generate 3 variants)
    for _ in range(3):
        neighbor = seed.copy()
        channel = random.randint(0, channels - 1)

        # Add smaller noise to whole channel
        channel_noise = np.random.uniform(
            -max_pixel_change * 0.3,
            max_pixel_change * 0.3,
            (height, width)
        )
        neighbor[:, :, channel] += channel_noise
        neighbors.append(neighbor)

    # Strategy 3: Channel-wise perturbation (generate 2 variants)
    for _ in range(2):
        neighbor = seed.copy()
        # Select a random channel
        channel = random.randint(0, channels - 1)

        # Generate small noise pattern for the entire channel
        channel_noise = np.random.uniform(
            -max_pixel_change * 0.3, max_pixel_change * 0.3,
            (height, width)
        )

        # Apply to channel and clip
        channel_data = neighbor[:, :, channel]
        new_channel = np.clip(channel_data + channel_noise, 0, 255)
        neighbor[:, :, channel] = new_channel
        neighbors.append(neighbor)

    # Strategy 4: Gaussian noise with small sigma (generate 2 variants)
    for _ in range(2):
        neighbor = seed.copy()
        noise = np.random.normal(0, max_pixel_change * 0.1, seed.shape)
        neighbor += noise
        neighbors.append(neighbor)

    return neighbors

# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """

    best_fitness = float('inf')
    best_candidate = None

    for candidate in candidates:
        fitness = compute_fitness(candidate, model, target_label)

        if fitness < best_fitness:
            best_fitness = fitness
            best_candidate = candidate

    return best_candidate, best_fitness


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """

    current_image = initial_seed.copy()
    current_fitness = compute_fitness(current_image, model, target_label)

    budget = 255 * epsilon
    min_bound = np.clip(initial_seed - budget, 0, 255)
    max_bound = np.clip(initial_seed + budget, 0, 255)

    print(f"Initial fitness: {current_fitness:.4f}")

    no_improvement_count = 0
    max_no_improvement = 20

    for iteration in range(iterations):
        # Generate neighbors
        raw_neighbors = mutate_seed(initial_seed, epsilon)

        valid_neighbors = []
        for n in raw_neighbors:
            n_clipped = np.clip(n, min_bound, max_bound)
            n_clipped = np.clip(n_clipped, 0, 255)
            valid_neighbors.append(n_clipped)

        # Add current image to candidates (elitism)
        candidates = valid_neighbors + [current_image.copy()]

        # Select best candidate
        best_candidate, best_fitness = select_best(candidates, model, target_label)

        # Check if we found improvement
        if best_fitness < current_fitness:
            current_image = best_candidate.copy()
            current_fitness = best_fitness
            no_improvement_count = 0

            print(f"Iteration {iteration}: New fitness = {current_fitness:.4f}")

            # Check if attack succeeded (target is not top prediction)
            if best_fitness < 0:  # Negative fitness means target is not top
                print(f"Attack successful at iteration {iteration}")
                return current_image, current_fitness
        else:
            no_improvement_count += 1

        # Early stopping if no improvement for too long
        if no_improvement_count >= max_no_improvement:
            print(f"Stopping early after {max_no_improvement} iterations without improvement")
            break

        # Print progress every 50 iterations
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Current fitness = {current_fitness:.4f}")

    print(f"Final fitness after {iterations} iterations: {current_fitness:.4f}")
    return current_image, current_fitness


# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Comment out if you want to use GPU
    model = vgg16.VGG16(weights="imagenet")

    epsilon = 0.30
    iterations = 300

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick first entry
    item = image_list[9]
    image_path = "images/" + item["image"]
    target_label = item["label"]

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original image")
    plt.axis('off')

    img_array = img_to_array(img)
    seed = img_array.copy()

    # Print baseline top-5 predictions
    print("\nBaseline predictions (top-5):")
    preds = model.predict(np.expand_dims(seed, axis=0))
    for cl in decode_predictions(preds, top=5)[0]:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    # Run hill climbing attack
    start_time = time.time()
    final_img, final_fitness = hill_climb(
        initial_seed=seed,
        model=model,
        target_label=target_label,
        epsilon=epsilon,
        iterations=iterations
    )
    end_time = time.time()

    print(f"\nAttack completed in {end_time - start_time:.2f} seconds")
    print("Final fitness:", final_fitness)

    plt.subplot(1, 2, 2)
    plt.imshow(array_to_img(final_img))
    plt.title(f"Adversarial Result\nfitness={final_fitness:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Print final predictions
    final_preds = model.predict(np.expand_dims(final_img, axis=0))
    print("\nFinal predictions (top-5):")
    decoded_final = decode_predictions(final_preds, top=5)[0]
    for cl in decoded_final:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    # Calculate perturbation
    perturbation = np.abs(final_img - seed)
    max_perturbation = np.max(perturbation)
    avg_perturbation = np.mean(perturbation)

    base_results_folder = "hc_results"
    timestamp = int(time.time())

    results_folder = os.path.join(base_results_folder, f"{target_label}_{timestamp}")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    array_to_img(seed).save(os.path.join(results_folder, "original.png"))
    array_to_img(final_img).save(os.path.join(results_folder, "adversarial.png"))

    info = {
        "target_label": target_label,
        "final_fitness": float(final_fitness),
        "mutation_type": "mixed_strategies",
        "top_prediction": decoded_final[0][1],
        "top_confidence": float(decoded_final[0][2]),
        "perturbation": {
            "max_pixel_change": f"{max_perturbation:.2f} (allowed: {255*0.30:.2f})",
            "average_pixel_change": f"{avg_perturbation:.2f}",
            "L_inf_distance": f"{np.max(np.abs(final_img - seed)):.2f}",
        },
        "epsilon": epsilon,
        "iterations": iterations,
        "success": bool(final_fitness < 0),
        "time_taken": f"{end_time - start_time:.2f} seconds",
    }

    with open(os.path.join(results_folder, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"\n[INFO] Saved results to unique folder: {results_folder}/")