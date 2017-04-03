import numpy as np
import copy
import statistics as stat
import matplotlib.pyplot as plt

#seed random number to make calculations consistent
#np.random.seed(1)

class Neuron:
    def __init__(self, name, layer, number, neuron_type="normal"):
        self.name = name
        self.layer = layer
        self.number = number
        self.neuron_type = neuron_type
        self.input_branches = []
        self.output_branches = []
        self.value = None
        self.error = None

    def __repr__(self):
        return "(Neuron Object. Layer {}, Number {}. Value {}.)".format(self.layer, self.number, self.value)

class Branch:
    def __init__(self, start, end, weight=float(0)):
        self.start_neuron = start
        self.end_neuron = end
        self.weight = weight
        self.stored_data = None

    def __repr__(self):
        return "(Branch from ({}, {}) to ({}, {}), Weight {})".format(
            self.start_neuron.layer,
            self.start_neuron.number,
            self.end_neuron.layer,
            self.end_neuron.number,
            self.weight)

class Network:
    def __init__(self, input_count, output_count, hidden_layers_count, hidden_layers_neuron_count, random_weights=True, bias_node=True):
        self.input_count = input_count
        self.output_count = output_count
        self.hidden_layers_count = hidden_layers_count
        self.hidden_layers_neuron_count = hidden_layers_neuron_count
        self.neurons = []
        self.initialise_neurons()
        self.initialise_branches()
        self.branches = [[y.input_branches for y in x] for x in self.neurons]
        self.initialise_bias(bias_node)
        if random_weights: self.randomise_branch_weights()

    def propagate(self, data):
        if len(data) != len(self.neurons[0]):
            raise AttributeError("Wrong number of data points provided. Expected {}, got {}".format(len(neurons[0]), len(data)))
        for i in range(len(data)):
            self.neurons[0][i].value = data[i]
        for i in range(1, len(self.neurons)):
            current_layer = self.neurons[i]
            for neuron in current_layer:
                total = 0
                for branch in neuron.input_branches:
                    total += branch.start_neuron.value * branch.weight
                neuron.value = self.stolen_sigmoid_function(total)

    def train_network_back_propagate(self, data, repititions):
        # for i in range(repititions):
        #     for example in data:
        #         self.propagate(example[0])
        #         result = [x.value for x in self.neurons[-1]]
        #         error = [x-y for x, y in zip(result, data[1])]
        #         error_slopes = [self.stolen_sigmoid_function(x, True) for x in error]len(self.neurons)len(self.neurons) #delta output sums
        #
        #         for i in range(len(self.neurons[-1])):
        #             neuron = self.neurons[-1][i]               #Set Neuron
        #             error_slope = error_slopes[i]              #Set appropriate error slope
        #             for branch in neuron.input_branches:
        #                 branch.stored_data = branch.weight + (branch.weight * error_slope))   #Work out new weight ands store it
        #
        #         for i in range(len(self.neurons)-2, 0, -1): #For each layer of neurons, working backwards
        #             for neuron in self.neurons[i]:  #For each neuron in that layer
        #                 pass
        pass

    def initialise_neurons(self):
        for layer in range(self.hidden_layers_count+2):
            self.neurons.append([])
            neuron_count = self.hidden_layers_neuron_count
            neuron_type = "normal"
            if layer == 0:
                neuron_type = "input"
                neuron_count = self.input_count
            if layer == self.hidden_layers_count+1:
                neuron_type = "output"
                neuron_count = self.output_count

            for number in range(neuron_count):
                self.neurons[layer].append(Neuron(
                    "Layer {} Neuron {}".format(layer, number),
                    layer,
                    number,
                    neuron_type))

    def initialise_branches(self):
        for i in range(len(self.neurons)-1):
            next_layer_count = len(self.neurons[i+1])
            layer = self.neurons[i]
            for neuron in layer:
                for n in range(next_layer_count):
                    target_neuron = self.neurons[i+1][n]
                    branch = Branch(neuron, target_neuron)
                    neuron.output_branches.append(branch)
                    target_neuron.input_branches.append(branch)

    def initialise_bias(self, bias_node):
        if not bias_node:
            self.bias_node = None
            return
        self.bias_node = Neuron("Bias Node", -1, -1, neuron_type="bias")
        self.bias_node.value = 1
        for layer in self.neurons:
            for neuron in layer:
                if neuron.neuron_type != "input":
                    branch = Branch(self.bias_node, neuron)
                    self.bias_node.output_branches.append(branch)
                    neuron.input_branches.append(branch)

    def randomise_branch_weights(self):
        for layer in self.branches:
            for neuron in layer:
                for branch in neuron:
                    branch.weight = (2*np.random.random(1)[0]) - 1

    def retrieve_neuron(self, layer, number):
        return self.neurons[layer][number]

    def stolen_sigmoid_function(self, x, deriv=False):
        """With thanks to Andrew Trask
            iamtrask.github.io/2015/7/12/basic-python-network"""
        if deriv == True:
            return x*(1-x)      # Returns the slope of the sigmoid graph at x
        return 1/(1+np.exp(-x)) # 1 over (1 plus e to the power of negative x)

    def print_input_output(self):
        print("input", end=": ")
        for neuron in self.neurons[0]:
            print(round(neuron.value, 2), end = ", ")
        print("Output", end=": ")
        for neuron in self.neurons[-1]:
            print(round(neuron.value, 2), end = ", ")
        print()

    def print_self(self):
        for layer in self.neurons:
            for neuron in layer:
                print("(Neuron {}, {}: Value: {})".format(neuron.layer, neuron.number, neuron.value), end=", ")
            print()

    def print_all(self):
        for layer in self.neurons:
            for neuron in layer:
                print("Name: ", neuron.name)
                print("input_branches", neuron.input_branches)
                print("output_branches", neuron.output_branches)
                print()

    def export_graph(self):
        fig = plt.gcf()
        ax = fig.gca()
        plt.axis([-2, 4, -2, 4])
        for layer in self.neurons:
            for neuron in layer:
                circle = plt.Circle((neuron.layer+0.1, neuron.number+0.05), 0.2, color="gray")
                ax.add_artist(circle)
                ax.annotate("{}".format(round(neuron.value, 2)), xy=(neuron.layer, neuron.number))
                for branch in neuron.input_branches:
                    x, y = (branch.start_neuron.layer, branch.end_neuron.layer),(branch.start_neuron.number, branch.end_neuron.number)
                    line = plt.Line2D(x, y)
                    ax.add_artist(line)
                    ax.annotate("{}".format(round(branch.weight, 2)), xy=(x[0]+0.2, y[0]-(y[1]/10)))

        plt.show()

class RandomTrainer:
    def __init__(self, network_specs, training_data):
        self.network_specs = network_specs
        self.training_data = training_data
        self.initial_network = self.generate_network()

    def generate_network(self):
        return Network(*self.network_specs)

    def clone_networks(self, network, copies):
        network_list = []
        for i in range(copies):
            network_list.append(copy.deepcopy(network))
        return network_list

    def modulate_weights(self, network, intensity=1):
        for layer in network.branches:
            for neuron in layer:
                for branch in neuron:
                    branch.weight += ((2*(np.random.random(1)[0]))-1)/intensity
                    if branch.weight > 1: branch.weight = 1
                    if branch.weight < -1: branch.weight = -1

    def train_randomly(self, repititions):
        best_network = self.initial_network
        for rep in range(repititions):
            network_list = self.clone_networks(best_network, 5)
            for i in range(1, len(network_list)):
                self.modulate_weights(network_list[i], 2)
            errors = []
            for network in network_list:
                error_data = []
                for data in self.training_data:
                    network.propagate(data[0])
                    result = [x.value for x in network.neurons[-1]]
                    squared_errors = [(x - y)**2 for x, y in zip(result, data[1])]
                    root_mean_error = stat.mean(squared_errors)**0.5
                    error_data.append(root_mean_error)
                errors.append(stat.mean(error_data))
            best_network = network_list[errors.index(min(errors))]
            if rep % 100 == 0:
                print("rep {}".format(rep))
        return best_network




xor_switch_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
    ]

ran = RandomTrainer([2, 1, 2, 10], xor_switch_data)
network = ran.train_randomly(1000)
network.propagate([0, 0])
network.print_input_output()
network.propagate([1, 0])
network.print_input_output()
network.propagate([0, 1])
network.print_input_output()
network.propagate([1, 1])
network.print_input_output()

if input("Show Graph?") == "y":
    network.export_graph()
