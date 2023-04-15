class Neural_Network {
	constructor(input_layer_nodes_qty, hidden_layers_qty, output_layer_nodes_qty, hidden_layers_nodes) {
		this.input_layer_nodes_qty = input_layer_nodes_qty;
		this.hidden_layers_qty = hidden_layers_qty;
		this.output_layer_nodes_qty = output_layer_nodes_qty;
		this.hidden_layers_nodes = hidden_layers_nodes;

		this.weights = {
			input_layer: this.setWeights(input_layer_nodes_qty, hidden_layers_qty ? hidden_layers_nodes[0] : output_layer_nodes_qty),
			hidden_layers: (() => {
				const layers = {}

				for (let i = 0; i < hidden_layers_qty; i++) {
					layers['layer'+i] = this.setWeights(hidden_layers_nodes[i], hidden_layers_nodes[i+1] ? hidden_layers_nodes[i+1] : output_layer_nodes_qty)
				}

				return layers
			})(),
			biases: (() => {
				const biases = {
					bias0: this.setBiases(hidden_layers_qty ? hidden_layers_nodes[0] : output_layer_nodes_qty),
					bias_last: this.setBiases(output_layer_nodes_qty)
				}

				for (let i = 1; i < hidden_layers_qty; i++) {
					biases['bias'+i] = this.setBiases(hidden_layers_nodes[i])
				}

				return biases
			})()
		}
	}

	sigmoid(x) {
		return 1 / (1 + Math.exp(-x));
	}

	setWeights(layer_1, layer_2) {
		const weights = [];
		for (let i = 0; i < layer_1; i++) {
			const row = [];

			for (let j = 0; j < layer_2; j++) {
				row.push(Math.random() * 2 - 1);
			}

			weights.push(row);
	 }

	 return weights;
	}

	setBiases(nodes) {
		const bias = [];

		for (let i = 0; i < nodes; i++) {
			bias.push(Math.random() * 2 - 1);
		}

		return bias;
	}

	predict(input, hidden_output_returning = false) {
	 	const inputResults = []

		const nodes = this.hidden_layers_qty ? this.hidden_layers_nodes[0] : this.output_layer_nodes_qty
	 	for (let i = 0; i < nodes; i++) {
			let sum = 0
			for (let j = 0; j < this.input_layer_nodes_qty; j++) {
				sum += input[j] * this.weights.input_layer[j][i]
			}
			sum += this.weights.biases.bias0[i]
			inputResults.push(this.sigmoid(sum))
	 	}

		if (!this.hidden_layers_qty) {
			return inputResults
		}

		const hidden_results = [inputResults]
	 	let currentResult = [...inputResults]

	 	if (this.hidden_layers_qty > 1) {
		 	for (let layer = 1; layer < this.hidden_layers_qty; layer++) {
				let nextResult = []
				for (let i = 0; i < this.hidden_layers_nodes[layer]; i++) {
					let sum = 0;
					for (let j = 0; j < currentResult.length; j++) {
				 		sum += currentResult[j] * this.weights.hidden_layers['layer'+(layer-1)][j][i]
					}	
					sum += this.weights.biases['bias'+layer][i]
					nextResult.push(this.sigmoid(sum));
				}
				currentResult = [...nextResult]
				hidden_results.push(nextResult)
			}
		}
	
		const preOutput = [...currentResult]

		const output = [];

		for (let i = 0; i < this.output_layer_nodes_qty; i++) {
			let sum = 0;
			for (let j = 0; j < preOutput.length; j++) {
				sum += preOutput[j] * this.weights.hidden_layers['layer'+(this.hidden_layers_qty-1)][j][i]
			}
			sum += this.weights.biases.bias_last[i]
			output.push(this.sigmoid(sum));
		}

		if (hidden_output_returning) {
			return [output, hidden_results]
		}

		return output
	}
	
	train(inputData, expectations, epochs, learningRate = 0.1) {
		for (let epoch = 0; epoch < epochs; epoch++) {
			for (let singleInputData = 0; singleInputData < inputData.length; singleInputData++) {
				const preOutput = this.predict(inputData[singleInputData], true)
				const output = preOutput.length > 1 ? preOutput[0] : preOutput
				const hidden_output = preOutput.length > 1 ? preOutput[1] : null

				const outputErrors = []
				for (let i = 0; i < output.length; i++) {
					outputErrors.push(expectations[singleInputData][i] - output[i])
				}

				if (this.hidden_layers_qty > 0) {
					const firstHiddenErrors = []
					
					for (let i = 0; i < this.hidden_layers_nodes[this.hidden_layers_qty-1]; i++) {
						let sum = 0;
						for (let j = 0; j < this.output_layer_nodes_qty; j++) {
							sum += outputErrors[j] * this.weights.hidden_layers['layer'+(this.hidden_layers_qty-1)][i][j]
						}
						firstHiddenErrors.push(sum);
					}
					
					let currentResult = [...firstHiddenErrors]
					const allHiddenLayersErrors = [firstHiddenErrors]
					
					for (let hidden_layer = this.hidden_layers_qty-2; hidden_layer >= 0; hidden_layer--) {
						const hiddenErrors = [];
						
						for (let i = 0; i < this.hidden_layers_nodes[hidden_layer]; i++) {
							let sum = 0;
							for (let j = 0; j < currentResult.length; j++) {
								sum += currentResult[j] * this.weights.hidden_layers['layer'+hidden_layer][i][j];
							}
							hiddenErrors.push(sum);
						}
						currentResult = [...hiddenErrors]
						allHiddenLayersErrors.push(hiddenErrors)
					}
					allHiddenLayersErrors.reverse()
				
					for (let i = 0; i < this.output_layer_nodes_qty; i++) {
						for (let j = 0; j < this.hidden_layers_nodes[this.hidden_layers_qty-1]; j++) {
							const gradient = output[i] * (1 - output[i]) * outputErrors[i] * hidden_output[hidden_output.length-1][j];
							this.weights.hidden_layers['layer'+(this.hidden_layers_qty-1)][j][i] += learningRate * gradient;
						}
						const gradient = output[i] * (1 - output[i]) * outputErrors[i];
						this.weights.biases.bias_last[i] += learningRate * gradient;
					}
					
					for (let layer = this.hidden_layers_qty-2; layer >= 0; layer--) {
						for (let i = 0; i < this.hidden_layers_nodes[layer+1]; i++) {
							for (let j = 0; j < this.hidden_layers_nodes[layer]; j++) {
								const gradient = hidden_output[layer+1][i] * (1 - hidden_output[layer+1][i]) * allHiddenLayersErrors[layer+1][i] * hidden_output[layer][j];
								this.weights.hidden_layers['layer'+layer][j][i] += learningRate * gradient;
							}
							const gradient = hidden_output[layer+1][i] * (1 - hidden_output[layer+1][i]) * allHiddenLayersErrors[layer+1][i];
							this.weights.biases['bias'+(layer+1)][i] += learningRate * gradient;
						}	
					}
	
					for (let i = 0; i < this.hidden_layers_nodes[0]; i++) {
						for (let j = 0; j < this.input_layer_nodes_qty; j++) {
							const gradient = hidden_output[0][i] * (1 - hidden_output[0][i]) * allHiddenLayersErrors[0][i] * inputData[singleInputData][j]
							this.weights.input_layer[j][i] += learningRate * gradient;
						}
						const gradient = hidden_output[0][i] * (1 - hidden_output[0][i]) * allHiddenLayersErrors[0][i];
						this.weights.biases.bias0[i] += learningRate * gradient;
					}	
				} else {
					for (let i = 0; i < this.output_layer_nodes_qty; i++) {
						for (let j = 0; j < this.input_layer_nodes_qty; j++) {
							const gradient = output[i] * (1 - output[i]) * outputErrors[i] * inputData[singleInputData][j]
              this.weights.input_layer[j][i] += learningRate * gradient;
						}
            const gradient = output[i] * (1 - output[i]) * outputErrors[i];
            this.weights.biases.bias0[i] += learningRate * gradient;
          }
				}
			}
		}
	}
}

module.exports = Neural_Network