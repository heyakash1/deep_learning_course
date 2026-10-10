# Learning Notes: Fruit Network (PyTorch basics)

Personal reference notes for `04_fruit_network`. They explain what each part of the data pipeline, the model and the training loop does, and why it is there.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Key terms](#2-key-terms)
3. [The data pipeline](#3-the-data-pipeline-datapy)
4. [The model](#4-the-model-modelpy)
5. [Shape trace](#5-shape-trace)
6. [Training and backpropagation](#6-training-and-backpropagation-trainpy)
7. [Seeing backprop happen](#7-seeing-backprop-happen-demo_gradientspy)
8. [Experiments: normalization and learning rate](#8-experiments-normalization-and-learning-rate)
9. [Evaluation, checkpoints and tests](#9-evaluation-checkpoints-and-tests)
10. [Mistakes I hit along the way](#10-mistakes-i-hit-along-the-way)
11. [What comes next](#11-what-comes-next)

---

## 1. The big picture

Image files go in, class scores come out, and then the training loop uses those scores to improve the weights.

```
 image files on disk
        │  ImageFolder   find the files, read labels from folder names
        ▼
 one (image, label) at a time
        │  transform     resize → grayscale → tensor → normalize
        ▼
 tensors with values -1.0 – 1.0
        │  DataLoader    group into batches of 32, shuffle
        ▼
 images (32, 1, 20, 20)  +  labels (32,)
        │  FruitNetwork  flatten → 3 × [Linear + ReLU] → Linear
        ▼
 logits (32, 10)         one raw score per fruit class, per image
        │  CrossEntropyLoss(logits, labels)
        ▼
 loss                    one number: how wrong this batch was
        │  loss.backward()
        ▼
 gradients               how the loss changes if each weight changes
        │  optimizer.step()
        ▼
 weights nudged so the loss goes down
        │
        └── repeat for every batch, for every epoch
```

---

## 2. Key terms

| Term | Meaning |
|------|---------|
| **Batch** | A small group of examples (here 32) processed together as one unit. |
| **Epoch** | One full pass through every batch in the dataset. |
| **Tensor** | A multi-dimensional array that PyTorch can do math on and track gradients for. |
| **Transform** | A recipe applied to each image as it is loaded (resize, grayscale, convert, normalize). |
| **Normalization** | Shifting and scaling input values to a friendlier range (here, pixels from [0, 1] to [-1, 1]). |
| **Layer** (`nn.Linear`) | `w · x + bias`, the same idea as the MP neuron and the perceptron, with the weights stored as a matrix. |
| **Activation function** | A nonlinear function applied after a layer (here ReLU). Without it, stacked layers collapse into one. |
| **Forward pass** | Running an input through the network to get outputs. |
| **Logits** | The raw, unbounded scores from the last layer, before any softmax. |
| **Loss** | A single number measuring how wrong the network's outputs are. Training tries to make it small. |
| **Gradient** | For one weight: how much the loss changes if that weight increases a little. |
| **Backpropagation** | Computing the gradient of the loss for every weight, working backward from the loss through the layers. |
| **Optimizer** | The rule that uses the gradients to update the weights (here, SGD). |
| **Learning rate (`lr`)** | The step size of each weight update. |
| **Seed** | A fixed starting point for the random number generator, so a run can be repeated exactly. |
| **Gradient norm** | One number summarizing how large a layer's gradients are: the square root of the sum of squares of all its gradient entries. |
| **Dead ReLU** | A hidden neuron whose output is 0 for every input in a batch. It passes no gradient back, so its incoming weights don't change that step. |
| **Test set** | Images the network never trained on, used only to measure how well it handles new data. |
| **Accuracy** | The fraction of images whose predicted class matches the true class. |
| **Overfitting** | Doing much better on the training images than on new ones, because the network has memorized details of the training set. |
| **Checkpoint** | A saved file holding the trained weights, plus the settings needed to use them. |
| **`nn.Module`** | PyTorch's base class for anything with learnable parameters. |

### Why batches?

| Approach | Problem |
|----------|---------|
| One image at a time | Slow (the hardware sits mostly idle), and each update is based on a single noisy example. |
| All images at once | Huge memory use, and only one weight update per epoch, so learning is slow. |
| **Batches of 32** | A good compromise: efficient, with many updates per epoch. |

`batch_size=32` is a knob, not a law. 16, 64 and 128 are all common, and the right value is usually found by trial.

---

## 3. The data pipeline (`data.py`)

All the loading code lives in one function, `get_loaders()`, so nothing runs when another file imports it.

```python
def get_loaders(data_dir="data", batch_size=BATCH_SIZE):
    transform = transforms.Compose([...])
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return (train_loader, test_loader, train_data.classes)
```

### 3.1 `transforms.Compose`

```python
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

**What it does:** builds a pipeline. Nothing runs yet. It says "when an image passes through, do these steps in order".

| Step | Effect |
|------|--------|
| `Resize((20, 20))` | Every image becomes 20×20 pixels. The size goes in as one tuple. |
| `Grayscale(1)` | 3 colour channels become 1, which shrinks the input and drops colour information this exercise doesn't need. |
| `ToTensor()` | Converts the image (a PIL object) into a tensor and rescales pixels from integers 0–255 to floats 0.0–1.0. |
| `Normalize((0.5,), (0.5,))` | Computes `(pixel - 0.5) / 0.5`, which maps [0, 1] to [-1, 1]. The values are 1-tuples because there is one channel. |

**The `normalize` flag.** `get_loaders(normalize=True)` builds the steps in a plain Python list, appends `Normalize` only when the flag is on, and then wraps the list in `transforms.Compose(steps)`. `ImageFolder` needs the *composed* transform, because a bare list can't be called on an image. The default `True` keeps `train.py`, the demo and the evaluation on the normalized pipeline, and `experiments.py` can switch it off to compare.

### 3.2 `ImageFolder`

**What it does:** walks a folder and treats each subfolder name as a class label (`Apple 20`, `Banana 3`, ...). It builds an indexable dataset, where `train_data[i]` gives `(image, label)` with the transform applied.

**It is lazy.** It only records where the files are and how to load them, and does the actual work one image at a time, when needed.

> **Easy mistake:** pointing `ImageFolder` at `data` instead of `data/train`. It then sees two subfolders, `train` and `test`, and treats *those* as the classes. You get 2 classes instead of 10, and no error. Build the paths with `os.path.join(data_dir, "train")`, which also handles Windows backslashes. Printing `len(classes)` catches this at once.

### 3.3 `DataLoader`

```python
DataLoader(train_data, batch_size=batch_size, shuffle=True)
```

`ImageFolder` hands out one image at a time. `DataLoader` groups them into batches.

**Why `shuffle=True` for training:** if the data were sorted (all apples, then all bananas, ...), the network would see long runs of one class and learn skewed patterns. Shuffling re-randomizes the order every epoch. The test loader doesn't need it.

**Use the parameter.** If the function takes `batch_size` but the body uses the constant `BATCH_SIZE`, the parameter does nothing.

### 3.4 `next(iter(...))`

```python
images, labels = next(iter(train_loader))
```

`train_loader` is iterable, but it is not itself an iterator. `iter(...)` creates the iterator first, and `next(...)` then pulls one batch from it. That is handy for a shape check without writing a full loop. Writing `next(train_loader)` directly raises a `TypeError`.

---

## 4. The model (`model.py`)

### 4.1 The class and its constructor

```python
class FruitNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
```

- **`nn.Module`** is the base class. Inheriting from it makes `net(images)` work, and lets the optimizer find all the weights automatically.
- **`super().__init__()`** must be the first line of `__init__`. It sets up the module's internal bookkeeping. Skipping it causes confusing errors.
- **Pass the sizes in.** `input_size` is `IMAGE_SIZE * IMAGE_SIZE` and `num_classes` is `len(classes)`, so changing the image size or the class list can't silently break the model.

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

**You call `net(images)`, not `net.forward(images)`.** PyTorch calls `forward` for you, plus some bookkeeping.

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

## 6. Training and backpropagation (`train.py`)

### 6.1 The loop

```python
def train(net, train_loader, epochs=20, lr=0.05):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()              # 1. clear old gradients
            outputs = net(images)              # 2. forward pass
            loss = criterion(outputs, labels)  # 3. measure how wrong
            loss.backward()                    # 4. backpropagation
            optimizer.step()                   # 5. update the weights
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    return epoch_losses
```

The same five actions run for every batch, in this order:

| # | Code | What it does |
|---|------|--------------|
| 1 | `optimizer.zero_grad()` | Clears the gradients left over from the previous batch. PyTorch *adds* new gradients to old ones by default, so skipping this mixes batches together. |
| 2 | `net(images)` | The forward pass: 32 images in, `(32, 10)` logits out. |
| 3 | `criterion(outputs, labels)` | Compares the logits with the true labels and returns one number, the loss. |
| 4 | `loss.backward()` | **Backpropagation.** Fills in a `.grad` on every weight and bias. |
| 5 | `optimizer.step()` | Updates every weight using its gradient. |

### 6.2 The loss: `CrossEntropyLoss`

It takes the raw logits and the true class numbers (`labels`), applies softmax internally to turn scores into probabilities, and penalizes the network by how little probability it gave the *correct* class. High confidence in the right answer gives a small loss, and high confidence in a wrong answer gives a large one.

> **Sanity check:** a network that is guessing among 10 classes gives each class about 10% probability, so the loss is `ln(10) ≈ 2.30`. The first epoch should start near that. If it starts at 50 or 0.01, something is wrong before any learning has happened.

### 6.3 Backpropagation in plain words

- **During the forward pass,** PyTorch quietly records every operation (this record is the *computation graph*, and the feature is called *autograd*).
- **`loss.backward()`** walks that record in reverse, from the loss back toward the input, applying the chain rule at each layer: "how does the loss change if this layer's output changes?", multiplied back through the layers.
- **The result** is a `.grad` on every parameter, with the same shape as the parameter. It answers: if this weight went up a little, would the loss go up or down, and by how much?

A positive gradient means increasing the weight would increase the loss, so the weight should go down. A negative gradient means the opposite.

### 6.4 The update: gradient descent

With plain SGD, `optimizer.step()` does this to every weight:

```
w  ←  w − lr × grad
```

Each weight moves a small step *against* its gradient, downhill on the loss. `lr` is the step size. (Strictly, computing the gradient on a batch of 32 and updating after each one is "mini-batch gradient descent". PyTorch calls the optimizer SGD.)

**The learning rate is a trade-off:**
- **Too small:** the loss falls slowly and needs many epochs.
- **Too large:** a step can overshoot a good region, and the loss jumps back up. Section 8 has a real example.

### 6.5 Smaller points

- **`loss.item()`** turns a one-value tensor into a plain Python float, so the running total doesn't keep tracking gradients.
- **`running_loss / len(train_loader)`** is the average loss per batch over the epoch. The last batch is usually smaller than 32, so this is a slight approximation, which is fine for watching a trend.
- **`net.parameters()`** hands the optimizer every weight and bias that `nn.Module` registered.
- **Seeds:** the initial weights and the shuffle order are random. `torch.manual_seed(...)` at the start of the script makes a run repeatable. Without it, two runs of the same code give different loss curves.
- **Don't train twice by accident.** `losses = train(...)` followed by `print(train(...))` trains the network a second time. Keep the result and print `losses`.

---

## 7. Seeing backprop happen (`demo_gradients.py`)

`train.py` runs backpropagation thousands of times without showing anything. This script runs it **once**, on one batch, and prints evidence at each stage.

### 7.1 What it proves

| Claim | Evidence printed |
|-------|------------------|
| Gradients don't exist until `backward()` runs | `.grad` is `None` before the backward pass |
| `backward()` fills in a gradient for every parameter, including the first layer | A non-zero gradient norm for each parameter, `fc1` included |
| The update rule is `w ← w − lr × grad` | The actual change of one weight, next to `-lr * grad` |
| The update moves the weights in the direction that lowers the loss | The loss on the same batch before and after the step |

### 7.2 The script, piece by piece

**Before the backward pass**

```python
optimizer.zero_grad()
print(net.fc4.weight.grad)   # None
```

`zero_grad()` clears the gradients. In PyTorch 2.0 and later it sets them to `None` instead of zeros by default. On a brand-new network no gradient has ever been computed, so `.grad` is `None` anyway. Either way, it shows that no gradient exists before `backward()`.

**Forward, loss, backward**

```python
outputs = net(images)
initial_loss = criterion(outputs, labels)
initial_loss.backward()
```

Steps 2 to 4 of the training loop. After `backward()`, every parameter has a `.grad` with the same shape as the parameter itself.

**Choosing which weight to watch**

```python
grad_abs = net.fc4.weight.grad.abs()
max_idx = divmod(torch.argmax(grad_abs).item(), grad_abs.shape[1])
```

- `.abs()` takes the absolute value of each gradient entry, since only the size matters for "which is largest".
- `torch.argmax(...)` returns the position of the largest entry as an index into the **flattened** tensor. `fc4.weight` has 10×10 = 100 entries, so that is a number from 0 to 99. `.item()` turns it into a plain Python int.
- `divmod(a, b)` returns `(a // b, a % b)`. With the flat index and the number of columns (`grad_abs.shape[1]`), that is exactly (row, column). For example, flat index 37 with 10 columns gives `(3, 7)`.
- `max_idx` is then a tuple of two ints, which prints readably and can index the weight directly: `weight[max_idx]`.
- `torch.unravel_index(flat, shape)` does the same job (PyTorch 2.2 and newer), but it returns a tuple of tensors, which prints less readably. `divmod` works on every version.
- `fc4.weight` has shape `(10, 10)`. `nn.Linear(10, num_classes)` stores its weights as (outputs, inputs), so the row is the output class and the column is the hidden neuron feeding it.

**Why the largest gradient, not a fixed index like `[0, 0]`:** ReLU can output 0 for a hidden neuron on every image in a batch. Such a "dead" neuron passes no gradient back, so the weights attached to it get a gradient of exactly 0 and don't change that step. A fixed index might land on one, and the demo would look like nothing happened.

**Reading the weight and its gradient**

```python
grad_val = net.fc4.weight.grad[max_idx].item()
initial_weight = net.fc4.weight[max_idx].item()
```

- `.item()` turns a one-value tensor into a plain Python float.
- **Copy the weight's value before the step.** The optimizer changes weights in place, so a stored reference would show the new value afterwards. A float is a snapshot.
- **Read the gradient from the parameter**, `net.fc4.weight.grad[...]`. Indexing the parameter first (`w = net.fc4.weight[0, 0]`, then `w.grad`) gives `None`: indexing creates a new tensor derived from the parameter, and the gradient is stored on the parameter itself, not on that derived tensor.

**Gradient norms across the layers**

```python
for name, param in net.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm().item())
```

- `named_parameters()` yields `(name, parameter)` pairs: `fc1.weight`, `fc1.bias`, and so on up to `fc4.bias`, eight in total.
- `grad.norm()` is the square root of the sum of the squares of all the gradient entries: one number summarizing how big that gradient is.
- **A non-zero norm on `fc1` is the real evidence of backprop.** `fc1` is the layer furthest from the loss, so its gradient had to travel back through `fc4`, `fc3` and `fc2` first.
- Compare "is it non-zero", not "which is bigger". A norm grows with the number of entries, so layers of different sizes aren't directly comparable.
- `if param.grad is not None` guards against a parameter that took no part in the loss.

**The update and the check**

```python
optimizer.step()
updated_weight = net.fc4.weight[max_idx].item()
actual_change = updated_weight - initial_weight
expected_change = -lr * grad_val
math.isclose(actual_change, expected_change, abs_tol=1e-7)
```

- SGD sets `w ← w − lr × grad`, so the change should be exactly `−lr × grad`.
- Weights are float32, which holds about 7 significant digits, so the two numbers can differ in the last few digits. Comparing with `==` would be unreliable.
- `math.isclose` uses a relative tolerance by default, which is useless for numbers near 0. `abs_tol` supplies a fixed tolerance for that case.

**Loss before and after**

```python
with torch.no_grad():
    updated_loss = criterion(net(images), labels)
```

- `torch.no_grad()` switches off gradient recording inside the block, so no computation graph is built. Use it whenever you only want a value, such as for evaluation.
- One step with a small learning rate normally lowers the loss on the same batch, but a single step isn't guaranteed to.

### 7.3 Small details

- `:.3e` is scientific notation (`1.234e-03`). Fixed decimals like `:.6f` can print a tiny number as `0.000000`.
- `torch.manual_seed(42)` fixes the initial weights and the shuffle order, so the printout is identical on every run.
- `net: nn.Module` and `images: torch.Tensor` are type hints. They are documentation, and PyTorch doesn't enforce them.
- `train_loader, _, classes = get_loaders()`: `_` is the convention for a value you don't use.
- An f-string with no `{}` inside doesn't need the `f`.
- The network's sizes come from `IMAGE_SIZE * IMAGE_SIZE` and `len(classes)`, as in `train.py`. That means `IMAGE_SIZE` has to stay imported.
- The script changes the network by one step. It is a demo, not part of training.
- The docstring says *what* the function does in one sentence. The comments say *why* (for example, "largest gradient, so a dead ReLU doesn't hide the update") and don't repeat what the code already says.
- A leading `\n` inside a printed string, as in `print("\n--- ... ---")`, prints a blank line first, which separates the sections of the output.

### 7.4 "One step updates all the weights": check it, don't assume it

`optimizer.step()` updates every parameter that has a gradient, not only the one being watched. But "all" is too strong: weights behind a dead ReLU have a gradient of exactly 0 and don't move. So the script counts instead of claiming:

```python
params_before = {name: param.detach().clone() for name, param in net.named_parameters()}

optimizer.step()

changed_params_count = 0
total_params_count = 0
for name, param in net.named_parameters():
    diff = param - params_before[name]
    changed_params_count += (diff != 0).sum().item()
    total_params_count += param.numel()
```

- The dictionary comprehension `{key: value for ...}` stores each parameter's snapshot under its name (`fc1.weight`, ...), so the same name finds it again after the step.
- `.detach()` cuts a tensor off from the computation graph.
- `.clone()` makes an independent copy. Without it, the snapshot would share memory with the parameter and change along with it, so every difference would be 0.
- `param - params_before[name]` is the element-wise difference, with the same shape as the parameter. `(diff != 0)` is a True/False tensor, `.sum()` counts the Trues, and `.item()` turns the count into a plain number.
- `param.numel()` is the number of entries in a tensor. Biases count too, so these are "parameter entries", not just weights.
- **The total for this network is 4340:** `fc1` has 400×10 + 10 = 4010 entries, and `fc2`, `fc3` and `fc4` have 10×10 + 10 = 110 each.

**Caveats**

- **Float32 rounding:** if `lr × grad` is far smaller than the weight itself (roughly below 1e-7 of it), adding it changes nothing, so the count of changed entries can fall slightly below the count of non-zero gradients. For an exact count of non-zero gradients, count `(param.grad != 0).sum()` as well and compare the two.
- **The count may come out close to the total or well below it.** Either result is informative: well below means some hidden neurons were dead for the whole batch.
- `param - params_before[name]` builds a small computation graph, because `param` requires gradients. That is harmless here. Using `param.detach()` or wrapping the loop in `torch.no_grad()` avoids it.

### 7.5 What a good run looks like

- `.grad` prints `None` before `backward()`.
- Every parameter has a non-zero gradient norm, `fc1` included. If a whole layer's norm is exactly 0, nothing flowed back, which points to a bug such as a detached tensor.
- The actual change equals `-lr * grad` to within rounding, and `isclose` prints `True`.
- The loss after the step is slightly lower than before.
- The final note reports how many parameter entries changed out of 4340.

### 7.6 My run (seed 42, `lr = 0.01`, one batch of 32)

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

**Reading it**

| Line | What it shows |
|------|---------------|
| `Gradient before backward(): None` | No gradient exists until `backward()` runs. |
| `Initial Loss: 2.303889` | Very close to `ln(10) = 2.3026`, the loss of a network guessing among 10 classes. Expected for random starting weights. |
| `Selected Weight Index: (0, 6)` | The entry of `fc4.weight` with the largest gradient: row 0 (output class 0, `Apple 20`), column 6 (hidden neuron 6). |
| `Gradient: -4.399e-02` | Negative, so raising this weight would lower the loss. |
| Eight non-zero norms | The gradient reached every layer, including `fc1`, the one furthest from the loss. |
| `Updated Weight: -8.380e-02` | The weight went from -0.08424 to -0.08380. It **increased**, against the sign of its negative gradient. |
| `Actual Change` = `Expected Change` | 4.399e-04 = −0.01 × (−4.399e-02). The update rule `w ← w − lr × grad` holds. |
| Loss 2.303889 → 2.303051 | The loss on the same batch fell by about 8.4e-4. |

**A check on the size of the loss drop**

For a small step, the loss should fall by about `lr × (sum of the squared gradients)`. The step moves every parameter by `−lr × grad`, and to first order the loss changes by the gradient multiplied by that movement. Squaring the eight printed norms and adding them gives 0.01568 + 0.00006 + 0.00032 + 0.00031 + 0.00105 + 0.00258 + 0.00899 + 0.05485 = 0.08384. Times `lr = 0.01`, that is 8.38e-4, which matches the observed drop (2.303889 − 2.303051 = 8.38e-4).

This only works because the step is small. For a large learning rate the loss surface curves enough that the first-order estimate fails, and the loss can go up instead of down. That is the same overshoot idea as the bumps in section 8.

**Things to be careful about**

- **Don't compare norms across layers.** `fc1.weight` has 4000 entries and `fc2.weight` has 100, so a larger norm for `fc1.weight` says little by itself.
- **4166 of 4340 entries changed, so 174 did not.** The script's note blames zero gradients, but the output doesn't prove that. One deduction is possible: a hidden neuron in `fc1` that was dead for the whole batch would leave 411 entries unchanged (its 400 weights, its bias and the 10 weights leaving it), which is more than 174. So no `fc1` neuron was fully dead. To find the real cause, print for each parameter how many entries have `grad == 0` and how many did not change, and compare the two.

---

## 8. Experiments: normalization and learning rate

Four runs of 20 epochs, changing one setting at a time:

| Normalize | lr | Epoch 1 | Lowest loss | Epoch 20 | What happened |
|-----------|----|---------|-------------|----------|---------------|
| no | 0.01 | 2.33 | 1.08 | 1.08 | Smooth but slow |
| no | 0.05 | 2.26 | 0.0465 (epoch 17) | 0.99 | Fast, then the loss jumped back up at epochs 18–20 |
| yes | 0.01 | 2.30 | 0.21 | 0.21 | Smooth, still falling at epoch 20 |
| yes | 0.05 | 2.17 | 0.0028 | 0.0028 | Fastest, with bumps at epochs 8 and 13 |

### What the runs show

- **Normalization helped at both learning rates.** The likely reason: Fruits-360 images have white backgrounds, so most raw pixels sit near 1.0. Every input to the first layer is then large and positive. Centering the pixels around 0 gives better-behaved gradients. This is the standard explanation, not something these runs prove.
- **A higher learning rate is faster but less steady.** The unnormalized lr 0.05 run reached 0.0465 and then jumped to 0.94: a step too large knocked the network out of a good region. The normalized lr 0.05 run had similar bumps but recovered each time.
- **The normalized lr 0.01 run is just slow.** It was still falling at epoch 20.

### Caveats

- **This is training loss, not accuracy.** A very low loss can mean the network learned the task, or that it memorized the training images. Only the test set can tell the two apart (`evaluate.py`).
- **The first row was run before seeding** (`torch.manual_seed`), so it is not an exact comparison with the other three. The other three share seed 20.
- **One run per setting.** This is suggestive, not proof. A different seed could place the bumps differently.
- **Choose settings from training behaviour,** then use the test set once at the end. Tuning on test accuracy would leak the test set into the choices.

### How the runs are produced (`experiments.py`)

```python
CONFIGS = [
    {"normalize": False, "lr": 0.01},
    {"normalize": False, "lr": 0.05},
    {"normalize": True, "lr": 0.01},
    {"normalize": True, "lr": 0.05},
]


def run_experiment(normalize, lr, epochs=20):
    torch.manual_seed(SEED)
    train_loader, _, classes = get_loaders(normalize=normalize)
    net = model.FruitNetwork(IMAGE_SIZE * IMAGE_SIZE, len(classes))
    return train(net, train_loader, epochs=epochs, lr=lr)
```

- **A list of dicts keeps the settings as data.** Adding a fifth experiment is one new line, with no new code.
- **`torch.manual_seed(SEED)` runs inside the function,** before anything random. Every run therefore starts from the same initial weights and sees the same shuffle order, so the runs differ only in the settings being compared. Seeding once at the top of the file would give each run a different starting point, because the random number generator would have moved on.
- **The order is seed, loaders, network, train.** That is the same order as the earlier standalone runs, so the three runs that had a fixed seed should reproduce their earlier numbers.
- **`get_loaders(normalize=normalize)`** is a keyword argument that switches `transforms.Normalize` on or off. `data.py` has to accept it and build its list of transforms conditionally.
- **`from train import train` is safe.** `train.py` only trains under `if __name__ == "__main__":`, so importing it runs nothing.
- **Results are stored in a dict with readable keys** such as `"norm=True, lr=0.05"`. `json.dump(results, f, indent=4)` writes it to `experiments.json`: JSON stores dicts and lists of numbers as plain text, and `indent=4` makes the file readable. `json.load` reads it back into the same dict.
- **`with open("experiments.json", "w") as f:`** closes the file automatically, even if an error happens inside the block.
- **Dicts keep insertion order,** so the curves come back in the order they were run.
- **`print_summary(results)`** prints the results as a Markdown table (loss at the first epoch, at the last epoch, and the lowest loss with its epoch), so it can be pasted straight into the README. `losses.index(lowest) + 1` gives the epoch of the lowest loss: the list index starts at 0 and epochs count from 1.
- Run it from inside `04_fruit_network`, because the file is created in the current folder.

### Plotting them (`plot_losses.py`)

```python
fig, (linear_ax, log_ax) = plt.subplots(1, 2, figsize=(14, 5))
for label, losses in results.items():
    epochs = range(1, len(losses) + 1)
    for ax in (linear_ax, log_ax):
        ax.plot(epochs, losses, label=label, marker="o", markersize=3)
log_ax.set_yscale("log")
fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.04))
fig.savefig(out_path, dpi=150, bbox_inches="tight")
```

- `plt.subplots(1, 2, figsize=(14, 5))` creates one figure with two side-by-side panels and returns the figure and both panels (called axes). Working with `fig` and `ax` objects is clearer than the `plt.` shortcuts once there is more than one panel.
- `range(1, len(losses) + 1)` numbers the epochs from 1. The list index starts at 0, so the x values have to be built separately.
- Each run is drawn on **both** panels. The linear panel shows the early drop. On a linear axis the late epochs would be squashed against the bottom, so the log panel (`set_yscale("log")`) gives equal ratios equal space, and the late-stage bumps stay visible.
- `ax.grid(True, which="both", linestyle="--", alpha=0.5)` draws dashed, half-transparent gridlines. `which="both"` includes the minor gridlines, which matter on a log axis.
- `MaxNLocator(integer=True)` keeps the x axis to whole epochs. Without it matplotlib picks ticks like 2.5 and 7.5.
- **One shared legend under both panels.** `linear_ax.get_legend_handles_labels()` returns the lines and their labels, and `fig.legend(...)` places them below the panels, so the legend never covers a curve.
- `fig.savefig(out_path, dpi=150, bbox_inches="tight")`: `dpi` is the resolution, and `tight` trims the margins while keeping the legend that sits outside the panels.
- **Save before show.** `show()` blocks until the window is closed, and once it closes the figure is gone, so a save after it writes a blank image.
- `show` is a parameter that defaults to `False`, so the function only opens a window when asked. `plt.close(fig)` at the end frees the figure's memory, which matters in loops and on machines with no display.
- It needs matplotlib: `pip install matplotlib`, and add it to `requirements.txt`.

---

## 9. Evaluation, checkpoints and tests

### 9.1 Why a separate test set

- **Training loss only measures fit to images the network has already seen.** A very low loss can mean it learned the task, or that it memorized the training images.
- **Test accuracy is measured on images it never trained on,** so it is an honest estimate of how it handles new data.
- **Test-set discipline:** choose the settings (normalization, learning rate) from the *training* curves, then run the test set once. If you pick the setting with the best test accuracy, the test set has leaked into your choices.
- **Compare train and test accuracy.** A large gap means overfitting.
- **A caveat about this dataset:** the Fruits-360 photos come from fruit being rotated, so many images look nearly identical. Test images can therefore resemble training images closely, and a high test accuracy may overstate how well the network would do on fruit photographed differently.

### 9.2 Saving and loading the weights (checkpoint)

```python
torch.save({
    "state_dict": net.state_dict(),
    "classes": classes,
    "normalize": normalize,
    "image_size": IMAGE_SIZE,
}, path)
```

- `net.state_dict()` is a dictionary from parameter names (`fc1.weight`, `fc1.bias`, ...) to tensors. It holds only the numbers, not the architecture.
- The checkpoint also stores the **settings** the weights depend on: the class names, whether the inputs were normalized, and the image size. Evaluation must preprocess images exactly as training did, and a mismatch (for example, training on normalized pixels and testing on raw ones) silently lowers the accuracy.
- **Loading:** build the same architecture first (`FruitNetwork(...)`), then call `net.load_state_dict(checkpoint["state_dict"])`.
- `torch.load(path, weights_only=True)` only loads plain data (tensors, dictionaries, lists, numbers, strings) and refuses arbitrary Python objects. That is safer, and it is the default in recent PyTorch.
- `load_network` raises a clear `ValueError` if the saved image size differs from `data.py`, and `evaluate.py` does the same if the class lists differ.
- The file is tiny (4340 numbers), so committing it is fine. Anyone can run `evaluate.py` without retraining.

### 9.3 `evaluate()` piece by piece

```python
net.eval()
with torch.no_grad():
    for images, labels in loader:
        predictions = net(images).argmax(dim=1)
        for label, prediction in zip(labels.tolist(), predictions.tolist()):
            total[label] += 1
            if prediction == label:
                correct[label] += 1
            else:
                confusions[(classes[label], classes[prediction])] += 1
```

- `net.eval()` switches the network to inference mode. It makes no difference here, because there is no dropout or batch norm, but it is the right habit. `net.train()` switches back.
- `torch.no_grad()` turns off gradient recording, since nothing is being trained.
- **`argmax(dim=1)`** returns, for each image, the index of its largest logit. That index is the predicted class. Softmax isn't needed, because it never changes which score is largest.
- `.tolist()` turns a tensor into a plain Python list, so it can be looped over with `zip`.
- `Counter` is a dictionary that returns 0 for a missing key, so `total[label] += 1` works the first time a class appears.
- **Accuracy** is `sum(correct) / sum(total)`, and **per-class accuracy** is `correct[i] / total[i]` for each class.
- **Confusions** record each mistake as a `(true class, predicted class)` pair. The most common ones show which fruits the network mixes up.
- `loader.dataset.targets` is the list of labels that `ImageFolder` stores for every image. `count_images` counts it, which gives the number of images per class for the README table.
- `f"{x:.1%}"` formats a fraction as a percentage with one decimal. The script prints a ready-to-paste Markdown table and saves `evaluation.json`.

### 9.4 `train.py` as a script

The final settings sit in constants at the top (`SEED`, `NORMALIZE`, `EPOCHS`, `LR`, `CHECKPOINT`), so they are in one place. Run as a script it seeds, builds the loaders, trains, and saves the checkpoint. The settings were chosen from the training-loss curves in section 8, not from test accuracy.

### 9.5 Tests (`test_model.py`)

- **Fake data, no dataset.** `torch.randn` makes random images and `torch.randint` makes random labels, with a seeded `torch.Generator`, so the tests are fast and reproducible on any machine.
- **A list can stand in for a loader.** `train` only loops over its loader and calls `len()` on it, so `[(images, labels)]` works as a loader with one batch.

| Test | What it checks |
|------|----------------|
| `test_output_shape` | A batch of 32 images gives an output of shape `(32, 10)`. |
| `test_one_step_changes_weights` | A gradient reaches `fc1` (the layer furthest from the loss), and one optimizer step changes its weights. |
| `test_loss_decreases` | Training repeatedly on one fixed batch lowers the loss. A network this size can overfit a single batch. |
| `test_evaluate_counts_correct_predictions` | `evaluate` reports accuracy 1.0 when the labels equal the predictions and 0.0 when every label is shifted, and records every mistake as a confusion. |

- **Testing both extremes** means the evaluate test can fail in either direction, as every test here should.

---

## 10. Mistakes I hit along the way

| Mistake | What goes wrong | Fix |
|---------|-----------------|-----|
| `def __int__` instead of `def __init__` | Python never runs it, so no layers are created and `forward` fails with `AttributeError`. | Spell it `__init__`. |
| Forgetting `super().__init__()` | The module's bookkeeping is never set up. | Make it the first line of `__init__`. |
| Data loading and prints at module level | They run every time another file imports this one. | Put them in functions, or under `if __name__ == "__main__":`. |
| `if __name__ == "__main":` | Missing the closing underscores, so the condition is never true and the script silently does nothing. | Write `"__main__"`. |
| Adding softmax before `CrossEntropyLoss` | Softmax is applied twice. | Return raw logits from `forward`. |
| `transforms.Resize(20, 20)` | The second argument is the interpolation mode, not a dimension. | Pass one tuple: `Resize((20, 20))`. |
| `ImageFolder("data")` | `train` and `test` become the classes: 2 classes, no error. | Join `data_dir` with `"train"` and `"test"`. |
| Ignoring the `batch_size` parameter | The caller's value does nothing. | Use the parameter in both `DataLoader` calls. |
| `iter(next(loader))` | `next` on a loader raises `TypeError`. | `next(iter(loader))`. |
| `import model` plus a `model` parameter | The parameter shadows the module, which leads to confusing errors. | Name the parameter `net`. |
| `m = model(images)` | Calls a module as a function, and would overwrite the network. | Call the network instance, not the module. |
| Calling `train(...)` twice | Trains the network a second time and prints the wrong list. | Keep `losses = train(...)` and print `losses`. |
| `model.Net()` | The class is `FruitNetwork`, so a wrong name raises `AttributeError`, and leaving out its two size arguments raises `TypeError`. | Use the real class name and pass `input_size` and `num_classes`. |
| Reading `.grad` from an indexed parameter, as in `net.fc4.weight[0, 0].grad` | Indexing makes a new tensor, and gradients live on the parameter itself, so the result is `None`. | Index the gradient instead: `net.fc4.weight.grad[0, 0]`. |
| Passing `normalize=` to `get_loaders` before adding that parameter | `TypeError: got an unexpected keyword argument`. | Add `normalize: bool = True` to `get_loaders` and build the list of transforms conditionally. |
| Calling `plt.savefig` after `plt.show()` | The figure is gone once the window closes, so the saved image is blank. | Save first, then show. |
| Passing a plain list of transforms to `ImageFolder` | `TypeError: 'list' object is not callable` when the first batch is loaded. | Wrap the list: `transforms.Compose(steps)`. |
| Evaluating with different preprocessing than the training used | Accuracy drops silently, with no error. | Save `normalize` in the checkpoint and read it back. |

---

## 11. What comes next

- **Run everything and paste the results** into the topic README: `train.py` (saves the weights), `evaluate.py` (accuracy and the per-class table), `experiments.py` and `plot_losses.py` (the table and the figure), and `test_model.py`.
- **Optional: the loss surface.** Save the weights after every epoch, project them to 2D with PCA, compute the loss on a grid in that plane, and draw the training path on top.
- **Optional experiments:** more epochs, or a learning rate between 0.01 and 0.05.