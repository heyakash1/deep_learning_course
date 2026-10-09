# Learning Notes: Fruit Network (PyTorch basics)

Personal reference notes for `04_fruit_network`. They explain what each part of the data pipeline, the model and the training loop does, and why it is there.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Key terms](#2-key-terms)
3. [The data pipeline](#3-the-data-pipeline-datapy)
4. [The model](#4-the-model-modelpy)
5. [Shape trace](#5-shape-trace)
6. [Training and backpropagation](#6-training-and-backpropagation-trainpy)
7. [Experiments: normalization and learning rate](#7-experiments-normalization-and-learning-rate)
8. [Mistakes I hit along the way](#8-mistakes-i-hit-along-the-way)
9. [What comes next](#9-what-comes-next)

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
- **Too large:** a step can overshoot a good region, and the loss jumps back up. Section 7 has a real example.

### 6.5 Smaller points

- **`loss.item()`** turns a one-value tensor into a plain Python float, so the running total doesn't keep tracking gradients.
- **`running_loss / len(train_loader)`** is the average loss per batch over the epoch. The last batch is usually smaller than 32, so this is a slight approximation, which is fine for watching a trend.
- **`net.parameters()`** hands the optimizer every weight and bias that `nn.Module` registered.
- **Seeds:** the initial weights and the shuffle order are random. `torch.manual_seed(...)` at the start of the script makes a run repeatable. Without it, two runs of the same code give different loss curves.
- **Don't train twice by accident.** `losses = train(...)` followed by `print(train(...))` trains the network a second time. Keep the result and print `losses`.

---

## 7. Experiments: normalization and learning rate

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

---

## 8. Mistakes I hit along the way

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

---

## 9. What comes next

- **Gradient demo:** print one weight, its `.grad`, and the weight after `optimizer.step()`, to show backprop and the update rule in action.
- **Loss curve plot:** matplotlib figure of the epoch losses, ideally with all four experiment runs overlaid.
- **Evaluation:** accuracy on the held-out test images, with the network in eval mode and gradients turned off.
- **Loss surface plot (optional):** save the weights each epoch, project them to 2D with PCA, and draw the loss landscape with the training path on top.