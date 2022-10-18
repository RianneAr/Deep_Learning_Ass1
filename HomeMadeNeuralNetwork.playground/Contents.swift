import Foundation

enum CustomError : Error {
    case inputOutputMismatch
    
}

class Node : Identifiable {
    
    enum NodeType {
        case Input, Hidden, Output
    }
    var type : NodeType
    var id = UUID()
    var neuralNetwork : NeuralNetwork?
    var parents : [Node]?
    var weights : [UUID : Double]
    var children : [Node]?
    var inputs : [UUID : Double]
    var output : Double?
    var setval : (UUID, Double)? {
        get {
            return nil
        }
        set {
            if let newValue = newValue {
                let parents = self.weights.keys
                if parents.contains(newValue.0) {
                    inputs[newValue.0] = newValue.1
                } else {
                    print("Unrecognised node sent \(newValue).")
                }
            }
            if inputs.count == weights.count {
                ActivateSignumFunction()
            }
        }
    }
    
    init(parents : [Node]?, nodeType : NodeType, for neuralNetwork : NeuralNetwork?, weights : [UUID : Double]) {
        self.type = nodeType
        if let parents = parents {
            self.parents = parents
        } else {
            print("Innitialised parentless node of type \(self.type).")
        }
        self.inputs = [:]
        self.neuralNetwork = neuralNetwork
        self.weights = weights
    }
    
    func setValue(value : (UUID, Double)) {
        self.setval = value
    }
    
    func AddChildren(children : [Node]?) -> Bool {
        if let children = children {
            self.children = children
            return false
        } else {
            print("Node \(self.id) got assigned 0 children.")
            return false
        }
    }
    
    func ActivateSignumFunction() -> Double {
//        print("Activation Function run in \(self.type) node")
        var result : Double = 0
        let e = 2.718281828459045
        if self.type == .Input {
            if let input = inputs.first {
                result = input.value
            }
        } else {
            if let parents = parents {                //Hidden or Output Node
                if inputs.count == parents.count {
                    for parent in parents {
                        result += (weights[parent.id]! * inputs[parent.id]!)      //May need fixing.
                    }
                    result += 1                                         // Bias weight
                    result = 1/(1 + powl(e, result))
                } else {
                    print("ActivationFunction Error: Input: \(inputs.count), parents \(parents.count) count mismatch")
                }
            }
        }
        if let children = children {
            for child in children {
                if let input = inputs.first {
                    child.setValue(value: (self.id, input.value))
                }
            }
        } else {
//            print("Final Output: \(result)")
            if let neuralNetwork =  neuralNetwork {
                neuralNetwork.output = result
            }
        }
        self.output = result
        return result
    }
    
    func UpdateWeights(previousDeltas : [UUID : Double], eta : Double, desiredOutput: Double) -> Double {
        var delta = 0.0
        let e = 2.718281828459045
        switch self.type {
        case .Input:
            self.inputs.removeAll()
        case.Output:
            var deltaW = eta
            if let output = output {
                delta = output * (1 - output) * (desiredOutput - output)
                for weight in weights {
                    deltaW = delta * eta * inputs[weight.key]!
                    self.weights[weight.key] = weight.value - deltaW
                }
            } else {
                print("Error, Update Weight before activating neuron.")
            }
            self.inputs.removeAll()
            return delta
        case .Hidden:
            if let children = children {
                let childrenID = children.map({$0.id})
                var deltaChild = 0.0
                if let output = output {
                    let childWeights = getChildrenWeights()
                    for child in childrenID {
                        deltaChild += (previousDeltas[child]! * childWeights[child]!)
                    }
                    delta = output * (1 - output) * deltaChild
                }
            } else {
                print("Error: Hidden node with no children.")
                return 0
            }
            for weight in weights {
                let deltaW = eta * delta * inputs[weight.key]!
                self.weights[weight.key] = weight.value - deltaW
                print("")
            }
            self.inputs.removeAll()
            return delta
        }
        return 0
    }
    
    func getChildrenWeights() -> [UUID : Double] {
        var childWeights = [UUID : Double]()
        if let children = children {
            for child in children {
                if let weight = child.weights[self.id] {
                    childWeights[child.id] = weight
                }
            }
            return childWeights
        } else {
            return [:]
        }
    }
    
}

class NeuralNetwork : Identifiable {
    var id = UUID()
    var output : Double = 0
    var networkHeads : [Node]
    var outputNode : [Node]?
    init(inputCount : Int, hiddenlayerWidth: Int, hiddenLayerDepth : Int, outputCount : Int) {
        self.networkHeads = [Node]()
        var currentLayer = [Node]()
        var parentLayer = [Node]()
        for _ in 1...inputCount {
            let node = Node(parents: nil, nodeType: .Input, for: self, weights: [self.id: 1])
            currentLayer.append(node)
        }
        networkHeads = currentLayer
        for _ in 1...hiddenLayerDepth {
            parentLayer = currentLayer
            currentLayer.removeAll()
            for _ in 1...hiddenlayerWidth {
                var weights = [UUID : Double]()
                for parent in parentLayer {
                    weights[parent.id] = 0.5
                }
                let node = Node(parents: parentLayer, nodeType: .Hidden, for: self, weights: weights)
                currentLayer.append(node)
            }
            for parent in parentLayer {
                parent.AddChildren(children: currentLayer)
            }
        }
        parentLayer = currentLayer
        currentLayer.removeAll()
        for _ in 1...outputCount {
            var weights = [UUID : Double]()
            for parent in parentLayer {
                weights[parent.id] = 0.5
            }
            let node = Node(parents: parentLayer, nodeType: .Output, for: self, weights: weights)
            currentLayer.append(node)
            
        }
        for parent in parentLayer {
            parent.AddChildren(children: currentLayer)
        }
        self.outputNode = currentLayer
    }
    
    func GetError(inputs: [(Double, Double)], output: [Double]) throws -> [Double] {
        guard inputs.count == output.count else {
            throw CustomError.inputOutputMismatch
        }
        var nnOutput = [Double]()
        var error = [Double]()
        for input in inputs {
            networkHeads[0].setValue(value: (self.id, input.0))
            networkHeads[1].setValue(value: (self.id, input.1))
            nnOutput.append(self.output)
        }
        for i in 0..<output.count {
            error.append((nnOutput[i] - output[i]) * (nnOutput[i] - output[i]))
        }
        return error
    }
    
    func BackTrack(learningRate : Double, desiredOutput : Double) {
        guard let outputNode = self.outputNode else {
            print("No output Nodes")
            return
        }
        var currentLayer = outputNode
        var previousDelta = [UUID : Double]()
        while !currentLayer.isEmpty {
            for node in currentLayer {
                let delta = node.UpdateWeights(previousDeltas: previousDelta, eta: learningRate, desiredOutput: desiredOutput)
                previousDelta[node.id] = delta
            }
            if let parents = currentLayer.randomElement()?.parents {
                currentLayer = parents
            } else {
                break
            }
        }
    }
    
    func Train(Epoches: Int, inputs: [(Double, Double)], Outputs: [Double], learningRate : Double) -> [Double] {
        var errors = [Double]()
        for _ in 1...Epoches {
            var temporaryErrors : Double = 0
            for i in 0..<inputs.count {
                networkHeads[0].setValue(value: (self.id, inputs[i].0))
                networkHeads[1].setValue(value: (self.id, inputs[i].1))
                temporaryErrors += (self.output - Outputs[i]) * (self.output - Outputs[i])
                BackTrack(learningRate: learningRate, desiredOutput: outputs[i])
                FlushNeuralNetwork()
            }
            temporaryErrors /= Double(Outputs.count)
            errors.append(temporaryErrors)
        }
        return errors
    }
    
    func FlushNeuralNetwork() {
        var nodes = self.outputNode
        while nodes != nil {
            if let nodes = nodes {
                for node in nodes {
                    node.inputs.removeAll()
                }
            }
            nodes = nodes?.first?.parents
        }
    }
}

let time1 = DispatchTime.now()
let nn = NeuralNetwork(inputCount: 2, hiddenlayerWidth: 2, hiddenLayerDepth: 1, outputCount: 1)
let time2 = DispatchTime.now()
let inputs : [(Double, Double)] = [(0.1, 0.1), (0.1, 1), (1, 0.1), (1, 1)]
let outputs : [Double] = [1, 1, 1, 1]
let errors = nn.Train(Epoches: 1024, inputs: inputs, Outputs: outputs, learningRate: 0.8)
let time3 = DispatchTime.now()
print("Execution Stats: \((time2.uptimeNanoseconds - time1.uptimeNanoseconds)/1_000_000) ms to make build neural network.")
print("Execution Stats: \((time3.uptimeNanoseconds - time2.uptimeNanoseconds)/1_000_000) ms to calculate error.")
print("Best performance \(errors.min()!)")
//let id = UUID()
//let node1 = Node(parents: nil, nodeType: .Input, for: nil, weights: [id : 1])
//let node2 = Node(parents: [node1], nodeType: .Hidden, for: nil, weights: [node1.id : 0.2])
//let node3 = Node(parents: [node2], nodeType: .Output, for: nil, weights: [node2.id : 0.4])
//node1.AddChildren(children: [node2])
//node2.AddChildren(children: [node3])
//for _ in 1...5000 {
//    node1.setValue(value: (id, 0.1))
//    let delta = node3.UpdateWeights(previousDeltas: [:], eta: 0.8, desiredOutput: 1)
//    let delta2 = node2.UpdateWeights(previousDeltas: [node3.id : delta], eta: 0.8, desiredOutput: 1)
//}
//print(node2.weights)
