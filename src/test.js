const Neural_Network = require('./Neural_network_library')

const nn = new Neural_Network(3, 2, 4, [16, 16]) // sum neural network

const trainingResults = nn.train([
          [0, 0, 0], 
          [1, 0, 0], 
          [1, 1, 0], 
          [0, 1, 1], 
          [0, 0, 1],
          [1, 0, 1], 
          [1, 1, 1]], 
          
          [[1, 0, 0, 0], 
           [0, 1, 0, 0], 
           [0, 0, 1, 0], 
           [0, 0, 1, 0], 
           [0, 1, 0, 0],
           [0, 0, 1, 0], 
           [0, 0, 0, 1]], 
           1000, 
           0.05)

// [1, 0, 0, 0] = 0
// [0, 1, 0, 0] = 1
// [0, 0, 1, 0] = 2
// [0, 0, 0, 1] = 3

console.log(trainingResults) // 100%
console.log(nn.predict([1, 0, 1])) // output ≈ [0, 0, 1, 0 ]