<h1 align="center"><b>Neural Network Library</b></h1>

# constructor

NeuralNetwork(**input_nodes_qty**, **hidden_layers_qty**, **output_nodes_qty**, **hidden_layers_nodes_qty**)

**input_nodes_qty** - type: number
**hidden_layers_qty** - type: number
**output_nodes_qty** - type: number
**hidden_layers_nodes_qty** - type: array of numbers

```javascript
  const nn = new NeuralNetwork(2, 3, 1, [4, 5, 3])
```
<hr>

# train

train(**input_data**, **target**, **epochs**, **learningRate** *= 0.1*)

**input_data** - type: array of arrays of numbers
**target** - type: array of arrays of numbers
**epochs** - type: number<br>
**learningRate** - type: float *by default **0.1***

```javascript
  const nnAccuracy = nn.train(
    [[0, 0], [1, 0], [0, 1], [1, 1]], // input_data
    [[0], [1], [0], [1]], // target
    50, // epochs
    0.1 // learningRate
  )

  // Accuracy of neural network after training
  console.log(nnAccuracy)
```
<hr>

# predict

predict(**input**)

**input** - type: array of numbers

```javascript
  nn.predict([0, 1])
```
<hr>

<h1 align="center"><b>Example</b></h1>

```javascript
  const nn = new NeuralNetwork(2, 3, 1, [4, 5, 3])

  const nnAccuracy = nn.train(
    [[0, 0], [1, 0], [0, 1], [1, 1]], // input_data
    [[0], [1], [0], [1]], // target
    50, // epochs
    0.1 // learningRate
  )

  // Accuracy of neural network after training
  console.log(nnAccuracy)

  nn.predict([0, 1]) // 0
```