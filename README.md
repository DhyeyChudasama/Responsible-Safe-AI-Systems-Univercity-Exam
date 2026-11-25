# Adversarial Attacks on an Automated Cheque-Amount Digit Reader

## University & Examination Details
* **University:** Charotar University of Science and Technology (CHARUSAT)
* **Institute:** Chandubhai S Patel Institute of Technology (CSPIT)
* **Program:** Seventh Semester of B. Tech (CE / IT / CSE / AIML)
* **Examination:** External Practical Examination (Set A)
* **Subject:** Responsible and Safe AI (`AIML4001`)
---

## Aim
To evaluate the vulnerability of a Convolutional Neural Network (CNN) based cheque-amount digit reader against **FGSM** and **PGD** adversarial evasion attacks, and to harden the model's robustness using **Adversarial Training**.

---

## Scenario Overview
A cooperative bank is deploying an automated cheque-clearing vision system to read handwritten digits. Malicious actors could introduce imperceptible pixel perturbations (e.g., via specially printed cheque backgrounds) causing the model to misclassify digits (such as reading a `1` as a `7` or a `3` as an `8`). This project simulates these evasion attacks across varying perturbation budgets (ε) and implements an adversarial defence mechanism prior to production deployment.

---

## Tech Stack & Environment Guidelines
* **Language:** Python 3
* **Libraries:** PyTorch, Torchvision, NumPy, Matplotlib
* **Dataset:** MNIST Handwritten Digits (`60,000` training / `10,000` testing images, `28×28` grayscale)
* **Execution Constraints:**
  * Random Seed: `torch.manual_seed(42)`
  * Device: `CPU`
  * Pixel Normalization Range: `[0, 1]`
  * Training Duration: Maximum `2 epochs` per training phase

---

## Algorithm & Methodology

1. **Baseline Model Training (Task 1):**
   * Load the MNIST dataset with pixel values scaled to `[0, 1]`.
   * Construct a CNN architecture with **2 Convolutional layers** and **2 Fully Connected layers**.
   * Train for `2 epochs` using the **Adam** optimizer (`lr = 0.001`, `batch_size = 64`) and record clean test accuracy.

2. **FGSM Evasion Attack (Task 2):**
   * Compute the gradient of the loss with respect to the input image: \(\nabla_x J(\theta, x, y)\).
   * Generate adversarial examples using: \(x_{adv} = \text{clip}(x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y)), 0, 1)\).
   * Evaluate baseline model accuracy for \(\epsilon \in \{0.05, 0.10, 0.20, 0.30\}\).

3. **PGD Evasion Attack (Task 3):**
   * Initialize perturbation randomly within \([-\epsilon, \epsilon]\).
   * Iteratively apply gradient ascent for `10 iterations` with step size α = ε / 4, projecting back onto the \(L_\infty\) ball of radius ε and clipping to `[0, 1]`.
   * Evaluate across \(\epsilon \in \{0.05, 0.10, 0.20, 0.30\}\) and compare against FGSM.

4. **Visual Inspection (Task 4):**
   * Generate a side-by-side comparison of 5 clean test digits and their FGSM adversarial counterparts (ε = 0.20) along with predicted labels.
   * Save output as `clean_vs_adversarial.png`.

5. **Adversarial Training Defence (Task 5):**
   * Fine-tune the baseline CNN for `2 epochs` using a **50:50 mixture** of clean images and FGSM adversarial images generated at ε = 0.20.
   * Re-evaluate clean, FGSM (ε = 0.20), and PGD (ε = 0.20) accuracies.

6. **Visualization & Comparison (Task 6):**
   * Plot **Accuracy vs. Epsilon (ε)** comparing FGSM and PGD degradation curves (`accuracy_vs_epsilon.png`).
   * Tabulate performance before and after adversarial training.

---

## Expected Output Format

> *Note: The values below represent the reference format. Record your exact observed values from your execution run.*

### Part A: Baseline Model under Attack
```text
Training baseline CNN (2 epochs)... done.
Clean test accuracy : 98.7 %

EPSILON   FGSM_ACC   PGD_ACC
-------------------------------
0.00      98.7 %     98.7 %
0.05      95.5 %     94.9 %
0.10      86.7 %     80.0 %
0.20      38.1 %      5.0 %
0.30       3.7 %      0.0 %
-------------------------------
Mis-read digits saved : clean_vs_adversarial.png
Attack success (eps=0.20): FGSM 61.9 %, PGD 95.0 %
```

### Part B: Adversarial Training (Defence)
```text
MODEL           CLEAN     FGSM(0.20)   PGD(0.20)
----------------------------------------------------
Baseline        98.7 %    38.1 %        5.0 %
Adversarially   99.1 %    92.1 %       73.1 %
trained
----------------------------------------------------
Clean accuracy change : +0.4 %
FGSM robustness gain  : +54.0 %
PGD gain              : +68.1 %
Plot saved : accuracy_vs_epsilon.png
```

---

## Technical Questions (Included in PDF Report)
1. **Tasks 1–2:** Report clean accuracy and FGSM accuracy across all $\epsilon$ values, identify the threshold where accuracy drops below 50%, and explain why minor perturbations cause misclassification in high-dimensional linear/piecewise-linear decision boundaries.
2. **Task 3:** Compare PGD and FGSM performance at identical $\epsilon$ values and explain why iterative multi-step optimization (PGD) outperforms single-step linearization (FGSM).
3. **Tasks 4–5:** Document the specific misclassified digits from `clean_vs_adversarial.png` and analyze how 50:50 adversarial training shifts clean, FGSM, and PGD accuracies.
4. **Task 6:** Define $\epsilon$ as the $L_\infty$ perturbation budget and propose two additional real-world defences for automated cheque clearing (e.g., *input preprocessing/JPEG compression/median filtering, defensive distillation, adversarial detection/rejection thresholds, or human-in-the-loop verification for high-value cheques*).

---
