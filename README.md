# SurFree Implementation:

Paper: https://arxiv.org/abs/2011.12807  
Original Github: https://github.com/t-maho/SurFree

--------

# What is SurFree?

SurFree is a fast, surrogate-free, decision-based black-box adversarial attack. It generates adversarial examples without relying on gradient estimation or surrogate models, operating solely with top-1 label queries. SurFree uses a geometry-aware directional search strategy, combining efficient binary searches with structured random perturbations to quickly find minimal-distortion adversarial examples. It achieves high success rates with significantly fewer queries compared to traditional decision-based attacks, making it practical for real-world black-box threat scenarios.

---------

# Basic Steps of the SurFree Algorithm

1. Start from a Correctly Classified Image: 
Begin with an input image 𝑥0 that the model classifies correctly.

2. Find Initial Boundary Point: 
Perform a binary search between 𝑥0 and a random misclassified point to find a point on the decision boundary (denoted x𝑏).

3. Generate Random Direction: 
Sample a random direction t in low frequency using DCT and generate the vecto v which is  orthogonalized against previous directions and u(unit vector from x0 to xb) for better exploration.


4. Sign Search: 
Move slightly along the random direction 𝑣 (both positive and negative) to determine which way reduces distortion and find new xa.

5. Update Boundary Point: 
Perform a binary search along the chosen direction to find a new boundary point closer to xo.


6. Optional Interpolation (Refinement): 
Use a quadratic interpolation step to better estimate the minimal distortion point, improving efficiency (optional but improves results).

7. Repeat Until Query Budget or Distortion Threshold Met are met to find the final adversarial.

-----------

# My contribution

In SurFree, the Discrete Cosine Transform (DCT) is used to transform the input image into a lower-dimensional frequency space. In the original SurFree paper, a full-image DCT is applied.
The main goal of using DCT is:

To focus on low-frequency components of the image,

And suppress high-frequency noise that may not contribute significantly to model decisions.

During each iteration, SurFree:

Constructs an orthogonal vector 𝑣k relative to a unit vector 𝑢𝑘
Forms a 2D plane with 𝑢𝑘 and 𝑣𝑘,
And searches for adversarial examples within this plane.

By emphasizing low-frequency directions, the attack becomes more query-efficient because low-frequency perturbations are more likely to affect the classifier meaningfully.

### Experimental Anlysis: Full DCT vs 8x8 DCT

In my experiments, I compared full DCT with 8×8 block DCT for SurFree:

Finding: 

Using 8×8 DCT increased both:

- The average L₂ distortion of the generated adversarial examples,
- The number of queries required to find adversarial samples.

### Reason for Increased Distrotion with 8x8 DCT 

When applying 8×8 DCT:

- The transformation is localized to small 8×8 patches,
- As a result, low- and high-frequency components are mixed within each block,
- This distorts the intended low-frequency emphasis.

Thus:

- The generated search directions are less clean,
- The adversarial search becomes less efficient,
- And both L₂ distortion and query budget increase compared to full DCT, where global low-frequency patterns are preserved.

----------


## Setup and Run SurFree Attack

Follow these steps to prepare the environment and run SurFree:

1. Clone the Repository

```bash
git clone https://github.com/t-maho/SurFree.git
cd SurFree
```

2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
# OR
venv\Scripts\activate      # For Windows
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Configuration for Customized Attack
Customize attack settings by modifying the config_example.json file.

DCT Type Selection:

For Full DCT:

```json
"dct_type": "full"
```

For 8×8 DCT:

```json
"dct_type": "8x8"
```

Attack Initialization Parameters:

Set maximum steps and query budget:

```json
"steps": 50,
"max_queries": 500
```

4. Running the Attack

```bash
python main.py
```
---------------------

# Models
The provided model is a pre-trained ResNet18 trained on the ImageNet dataset.

-------------------------------

### Requirements

```bash
torch==2.6.0
eagerpy==0.30.0
scipy==1.15.2
tqdm==4.67.1
scikit-learn==1.6.1
gitpython==3.1.44
requests==2.32.3
torchvision==0.21.0
```

