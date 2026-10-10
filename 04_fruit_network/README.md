# 04 – Fruit Network (Backpropagation)

Learning notes on the PyTorch pieces used here: see [NOTES.md](NOTES.md).

## What this is
A 4-layer fully connected network (10 neurons per layer) that classifies images of 10 kinds of fruit and vegetable. It is trained with PyTorch, where `loss.backward()` performs backpropagation automatically and an SGD optimizer applies the gradient descent updates.

Until now, every topic was written from scratch with a tiny truth table as the data. This one introduces a real image dataset, mini-batches, a loss function, and gradients computed by backpropagation.

## Architecture
```
400 inputs ──► 10 ──► 10 ──► 10 ──► 10 outputs
 (20×20)     ReLU   ReLU   ReLU   (logits, one per class)
```

- **Input:** a 20×20 grayscale image, flattened to 400 numbers.
- **Layers:** four `nn.Linear` layers, with ReLU after the first three. The last layer outputs raw scores (logits).
- **Loss:** `CrossEntropyLoss`, which applies softmax internally, so the network has no softmax layer.

## Dataset
[Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) by Mihai Oltean, `fruits-360_100x100` version, using 10 of its classes:

`Apple 20`, `Banana 3`, `Blueberry 1`, `Cantaloupe 1`, `Cucumber 3`, `Kiwi 1`, `Mango 1`, `Orange 1`, `Peach 1`, `Pineapple 1`

<!-- TODO: add the number of training and test images per class once counted -->

**Preprocessing:** resize to 20×20, convert to grayscale, scale pixels to [0, 1], then normalize to [-1, 1].

## Files
- `data.py`: `get_loaders()` builds the train and test `DataLoader`s (batches of 32, training data shuffled).
- `model.py`: the `FruitNetwork` class.
- `train.py`: the training loop (zero gradients, forward, loss, `backward()`, optimizer step).
- `demo_gradients.py`: runs backpropagation once on a single batch and prints the evidence: the gradient before and after `backward()`, one weight's update checked against `-lr × grad`, and the loss before and after.
<!-- TODO: add experiments.py, evaluate.py, test_model.py as they are written -->
- `NOTES.md`: personal learning notes explaining the code line by line.

## How to run
The images are not stored in this repo (`data/` is gitignored). To set them up:

1. Download the dataset from Kaggle and unzip it.
2. Copy the 10 class folders listed above from the `Training` and `Test` folders of `fruits-360_100x100` into `04_fruit_network/data/train/` and `04_fruit_network/data/test/`.

Then, from inside the `04_fruit_network` folder (the files import each other by name):

```
pip install -r ../requirements.txt
python data.py      # checks the loaders: shapes and class names
python train.py     # trains the network and prints the loss per epoch
python demo_gradients.py   # one backpropagation step, with the evidence printed
```

<!-- TODO: add the commands for the other scripts once they exist -->

## Training setup
- Optimizer: plain SGD, 20 epochs, batch size 32
- Seed: `torch.manual_seed(20)` for repeatable runs
- Chosen settings: <!-- TODO: fill in normalize yes/no and the final learning rate -->

## Experiments
Four 20-epoch runs, changing the normalization and the learning rate:

| Normalize | Learning rate | Loss at epoch 1 | Loss at epoch 20 | Notes |
|-----------|---------------|-----------------|------------------|-------|
| no  | 0.01 | 2.33 | 1.08   | Smooth but slow (run before seeding) |
| no  | 0.05 | 2.26 | 0.99   | Reached 0.047 at epoch 17, then jumped back up |
| yes | 0.01 | 2.30 | 0.21   | Smooth, still falling at epoch 20 |
| yes | 0.05 | 2.17 | 0.0028 | Fastest, with small bumps at epochs 8 and 13 |

**What this suggests:**
- Normalizing the inputs helped at both learning rates.
- A larger learning rate trains faster but less steadily.
- The first loss is close to `ln(10) ≈ 2.30`, which is what a 10-class network that is guessing should give.

These are training losses from single runs, so they show how training behaves, not how well the network classifies new images. The first row was run before the seed was fixed.

## Results
### Backpropagation on one batch
`demo_gradients.py` runs one forward and backward pass on a single batch of 32 images (seed 42, learning rate 0.01), then one optimizer step:

```
Gradient before backward(): None

--- Weight Update Demonstration ---
Initial Loss: 2.303889
Selected Weight Index: (0, 6)
Initial Weight: -8.424e-02
Gradient: -4.399e-02

--- Gradient Norms Across Layers ---
Layer 'fc1.weight' Gradient Norm: 1.252e-01
Layer 'fc1.bias' Gradient Norm: 7.987e-03
Layer 'fc2.weight' Gradient Norm: 1.793e-02
Layer 'fc2.bias' Gradient Norm: 1.750e-02
Layer 'fc3.weight' Gradient Norm: 3.245e-02
Layer 'fc3.bias' Gradient Norm: 5.082e-02
Layer 'fc4.weight' Gradient Norm: 9.480e-02
Layer 'fc4.bias' Gradient Norm: 2.342e-01

--- Post-Step Verification ---
Updated Weight: -8.380e-02
Actual Change: 4.399e-04
Expected Change (-lr * grad): 4.399e-04
Matches Expected Update (within tolerance): True

Loss Before Step: 2.303889
Loss After Step: 2.303051

Note: The optimizer step updated 4166 out of 4340 parameter entries (weights with non-zero gradients).
```

What it shows:
- **No gradient exists before `backward()`**, and afterwards every layer has a non-zero gradient, including `fc1`, the layer furthest from the loss. The gradient flowed back through the whole network.
- **The weight moved by exactly `-lr × grad`.** Its gradient was negative, so the weight increased.
- **The loss on the same batch fell** from 2.303889 to 2.303051. That drop matches the first-order prediction, `lr` times the sum of the squared gradient norms (0.01 × 0.0838 ≈ 8.4e-4).
- **The starting loss, 2.30, is `ln(10)`**, which is what a network guessing among 10 classes gives.
- **4166 of the 4340 parameter entries changed** in this one step.

<!-- TODO: loss curve plot: add the saved image, e.g. ![Loss curve](loss_curve.png) -->

<!-- TODO: test accuracy from evaluate.py, and which classes get confused -->

## Notes
- The loss is only meaningful alongside test accuracy: a very low training loss can also mean the network memorized the training images.
- The images are small and grayscale, and the network is a plain fully connected one, so this exercise is about showing backpropagation, not about reaching the best possible accuracy.