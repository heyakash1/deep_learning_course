# Learning Notes: Fruit Network (PyTorch basics)

Personal reference notes for `04_fruit_network`. They explain what each line of the data pipeline and the model does, and why it is there.

> These notes were written against the first version of `train.py`, where data loading and the model check lived together. The loading code is moving to `data.py`, but the concepts are unchanged.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Key terms](#2-key-terms)
3. [The data pipeline](#3-the-data-pipeline-trainpy)
4. [The model](#4-the-model-modelpy)
5. [Shape trace](#5-shape-trace)
6. [Mistakes I hit along the way](#6-mistakes-i-hit-along-the-way)
7. [What comes next](#7-what-comes-next)

---

## 1. The big picture

Everything so far turns image files on disk into 10 class scores per image. Nothing has learned yet; this is only the computation shape.

```
 image files on disk
        │  ImageFolder   find the files, read labels from folder names
        ▼
 one (image, label) at a time
        │  transform     resize → grayscale → tensor
        ▼
 tensors with values 0.0 – 1.0
        │  DataLoader    group into batches of 32, shuffle
        ▼
 images (32, 1, 20, 20)  +  labels (32,)
        │  FruitNetwork  flatten → 3 × [Linear + ReLU] → Linear
        ▼
 logits (32, 10)         one raw score per fruit class, per image
        │
        ▼
 next step: loss → backward → weight update
```

---

## 2. Key terms

| Term | Meaning |
|------|---------|
| **Batch** | A small group of examples (here 32) processed together as one unit. |
| **Epoch** | One full pass through every batch in the dataset. |
| **Tensor** | A multi-dimensional array that PyTorch can do math on and track gradients for. |
| **Transform** | A recipe applied to each image as it is loaded (resize, grayscale, convert). |
| **Layer** (`nn.Linear`) | `w · x + bias`, the same idea as the MP neuron and the perceptron, with the weights stored as a matrix. |
| **Activation function** | A nonlinear function applied after a layer (here ReLU). Without it, stacked layers collapse into one. |
| **Forward pass** | Running an input through the network to get outputs. |
| **Logits** | The raw, unbounded scores from the last layer, before any softmax. |
| **`nn.Module`** | PyTorch's base class for anything with learnable parameters. |

### Why batches?

| Approach | Problem |
|----------|---------|
| One image at a time | Slow (the hardware sits mostly idle), and each update is based on a single noisy example. |
| All images at once | Huge memory use, and only one weight update per epoch, so learning is slow. |
| **Batches of 32** | A good compromise: efficient, with many updates per epoch. |

`batch_size=32` is a knob, not a law. 16, 64 and 128 are all common, and the right value is usually found by trial.

---

## 3. The data pipeline (`train.py`)

### 3.1 `transforms.Compose`

```python
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
```

**What it does:** builds a pipeline. Nothing runs yet. It says "when an image passes through, do these three steps in order".

| Step | Effect |
|------|--------|
| `Resize((20, 20))` | Every image becomes 20×20 pixels. |
| `Grayscale(1)` | 3 colour channels become 1, which shrinks the input and drops colour information this exercise doesn't need. |
| `ToTensor()` | Converts the image (a PIL object) into a tensor and rescales pixels from integers 0–255 to floats 0.0–1.0, which networks train much better on. |

### 3.2 `ImageFolder`

```python
train_data = datasets.ImageFolder("data/train", transform=transform)
```

**What it does:** walks `data/train` and treats each subfolder name as a class label (`Apple 20`, `Banana 3`, ...). It builds an indexable dataset, where `train_data[i]` gives `(image, label)` with the transform applied.

**Worth knowing:** it is lazy. It only records where the files are and how to load them, and does the actual work one image at a time, when needed.

### 3.3 `DataLoader`

```python
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```

**What it does:** `ImageFolder` hands out one image at a time. `DataLoader` groups them into batches of 32.

**Why `shuffle=True`:** if the data were sorted (all apples, then all bananas, ...), the network would see long runs of one class and learn skewed patterns. Shuffling re-randomizes the order every epoch.

### 3.4 `next(iter(...))`

```python
images, labels = next(iter(train_loader))
```

`train_loader` is iterable: `for images, labels in train_loader:` hands out one batch at a time until the data runs out. `iter(...)` creates that iterator by hand, and `next(...)` pulls just the first batch. That is handy for a shape check without writing a full loop.

### 3.5 Building the model and running a batch

```python
m = model.FruitNetwork(input_size=400, num_classes=10)
outputs = m(images)
```

`FruitNetwork(...)` builds the network with randomly initialized weights, like the random `w` in the perceptron topic. `m(images)` runs the forward pass.

---

## 4. The model (`model.py`)

### 4.1 The class and its constructor

```python
class FruitNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
```

- **`nn.Module`** is the base class. Inheriting from it makes `m(images)` work, and lets the optimizer find all the weights automatically later.
- **`super().__init__()`** must be the first line of `__init__`. It sets up the module's internal bookkeeping (tracking layers and parameters). Skipping it causes confusing errors.

### 4.2 The layers

```python
self.fc1 = nn.Linear(input_size, 10)
self.fc2 = nn.Linear(10, 10)
self.fc3 = nn.Linear(10, 10)
self.fc4 = nn.Linear(10, num_classes)
```

Each `nn.Linear(a, b)` is one layer: `a` inputs feeding `b` neurons, with a weight on every connection and one bias per output neuron. Because the layers are assigned as attributes (`self.fc1`), `nn.Module` automatically registers their weights as things to learn.

```
400 inputs ──► 10 ──► 10 ──► 10 ──► 10 outputs
          fc1    fc2    fc3    fc4
```

### 4.3 ReLU

```python
self.relu = nn.ReLU()
```

ReLU is `max(0, x)` applied to each value independently: negatives become 0, positives pass through.

> **Why it matters:** without an activation, stacking linear layers is mathematically the same as one linear layer (a linear function of a linear function is still linear). ReLU adds the nonlinearity that lets a deeper network represent more than a single layer can. This is the same lesson as topic 03, where a hidden layer was needed to get past what one perceptron could do.

One `nn.ReLU()` instance can be reused in several places, because it has no learned parameters.

### 4.4 `forward`

```python
def forward(self, x):
    x = x.view(x.size(0), -1)

    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))

    x = self.fc4(x)
    return x
```

**You call `m(images)`, not `m.forward(images)`.** PyTorch calls `forward` for you, plus some bookkeeping.

| Line | What happens |
|------|--------------|
| `x.view(x.size(0), -1)` | Reshapes `(32, 1, 20, 20)` into `(32, 400)`. `nn.Linear` needs a flat list of numbers per example. `x.size(0)` keeps the batch dimension (32 images stay separate), and `-1` means "work this one out" (1×20×20 = 400). The data itself is unchanged. |
| `self.relu(self.fc1(x))` | Read inside-out: the linear layer runs first, then ReLU zeroes the negatives. The same pattern repeats for `fc2` and `fc3`. |
| `self.fc4(x)` | The final layer, with no ReLU after it. |
| `return x` | Returns `(32, 10)`: 10 raw scores per image. |

### 4.5 Why no activation on the last layer

The final layer returns **logits**, raw scores with no softmax. That is correct when training with `nn.CrossEntropyLoss`, which applies softmax internally (as `log_softmax` followed by negative log-likelihood).

> Adding your own softmax layer here as well would apply it twice, which makes training worse. If you ever want actual probabilities, for example "73% banana", call `torch.softmax(outputs, dim=1)` after training, on the logits. Never put it inside `forward` while training with `CrossEntropyLoss`.

---

## 5. Shape trace

Following one batch through the network:

| Step | Shape | What the dimensions mean |
|------|-------|--------------------------|
| Input batch | `(32, 1, 20, 20)` | batch, channels, height, width |
| After `view` | `(32, 400)` | batch, flattened pixels |
| After `fc1` + ReLU | `(32, 10)` | batch, hidden neurons |
| After `fc2` + ReLU | `(32, 10)` | batch, hidden neurons |
| After `fc3` + ReLU | `(32, 10)` | batch, hidden neurons |
| After `fc4` (logits) | `(32, 10)` | batch, one score per fruit class |

The last 10 is `num_classes`. It matches the hidden width only by coincidence.

When a shape error appears, it is almost always the flatten step.

---

## 6. Mistakes I hit along the way

| Mistake | What goes wrong | Fix |
|---------|-----------------|-----|
| `def __int__` instead of `def __init__` | Python never runs it, so no layers are created and `forward` fails with `AttributeError`. | Spell it `__init__`. |
| Forgetting `super().__init__()` | The module's bookkeeping is never set up. | Make it the first line of `__init__`. |
| Data loading and prints at module level | They run every time another file imports this one. | Put them in functions, or under `if __name__ == "__main__":`. |
| Adding softmax before `CrossEntropyLoss` | Softmax is applied twice. | Return raw logits from `forward`. |

---

## 7. What comes next

The training loop:

1. Zero the gradients.
2. Forward pass.
3. Compute the loss (`CrossEntropyLoss`).
4. `loss.backward()`, which is backpropagation.
5. Optimizer step (gradient descent).

Notes on those steps will be added here as they come up.