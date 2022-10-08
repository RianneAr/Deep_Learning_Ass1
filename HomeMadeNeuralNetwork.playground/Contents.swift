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
    var neuralNetwork : NeuralNetwork
    var parentsID : [UUID]?
    var weights : [UUID : Double]
    var children : [Node]?
    var inputs : [UUID : Double]
    var setval : (UUID, Double)? {
        get {
            return nil
        }
        set {
            if let newValue = newValue {
                if parentsID?.contains(newValue.0) ?? false {
                    inputs[newValue.0] = newValue.1
                } else {
                    print("Unrecognised node sent \(newValue).")
                }
            }
            if inputs.count == parentsID?.count {
                ActivateSignumFunction()
            }
        }
    }
    
    init(parents : [UUID]?, nodeType : NodeType, for neuralNetwork : NeuralNetwork, weights : [UUID : Double]) {
        self.type = nodeType
        if let parents = parents {
            self.parentsID = parents
//            print("Initiallised \(self.type) node with \(parents.count) parents.")
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
//            print("\(self.type) node \(self.id) got \(children.count) children.")
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
            if let parentsID = parentsID {                //Hidden or Output Node
                if inputs.count == parentsID.count {
                    for pid in parentsID {
                        result += (weights[pid]! * inputs[pid]!)      //May need fixing.
                    }
                    result += 1                                         // Bias weight
                    result = 1/(1 + powl(e, result))
                } else {
                    print("ActivationFunction Error: Input: \(inputs.count), parents \(parentsID.count) count mismatch")
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
            print("Final Output: \(result)")
            neuralNetwork.output = result
        }
        self.inputs.removeAll()
        return result
    }
}

class NeuralNetwork : Identifiable {
    var id = UUID()
    var output : Double = 0
    var networkHeads : [Node]
    init(inputCount : Int, hiddenlayerWidth: Int, hiddenLayerDepth : Int, outputCount : Int) {
        self.networkHeads = [Node]()
        var currentLayer = [Node]()
        var parentLayer = [Node]()
        for _ in 1...inputCount {
            let node = Node(parents: [self.id], nodeType: .Input, for: self, weights: [self.id: 1])
            currentLayer.append(node)
        }
        networkHeads = currentLayer
        var parentLayerID = [UUID]()
        for _ in 1...hiddenLayerDepth {
            parentLayer = currentLayer
            currentLayer.removeAll()
            parentLayerID = parentLayer.map({$0.id})
            for _ in 1...hiddenlayerWidth {
                var weights = [UUID : Double]()
                for uuid in parentLayerID {
                    weights[uuid] = 0.5
                }
                let node = Node(parents: parentLayerID, nodeType: .Hidden, for: self, weights: weights)
                currentLayer.append(node)
            }
            for parent in parentLayer {
                parent.AddChildren(children: currentLayer)
            }
        }
        parentLayer = currentLayer
        currentLayer.removeAll()
        parentLayerID = parentLayer.map({$0.id})
        for _ in 1...outputCount {
            var weights = [UUID : Double]()
            for uuid in parentLayerID {
                weights[uuid] = 0.5
            }
            let node = Node(parents: parentLayerID, nodeType: .Output, for: self, weights: weights)
            currentLayer.append(node)
            
        }
        for parent in parentLayer {
            parent.AddChildren(children: currentLayer)
        }
    }
    
    func UpdateWeights() -> Bool {
        var nodes = [[Node]]()
        if let head = self.networkHeads.first {
            var node : Node? = head
            while node?.type != .Output {
                if let layer = node?.children {
                    nodes.append(layer)
                    node = layer.randomElement() ?? nil
                }
            }
        }
        print(nodes.reversed())
        return false
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
    
}

let time1 = DispatchTime.now()
let nn = NeuralNetwork(inputCount: 2, hiddenlayerWidth: 2, hiddenLayerDepth: 1, outputCount: 1)
let time2 = DispatchTime.now()
let error = try? nn.GetError(inputs: [(0,0), (0,1), (1,0), (1,1)], output: [0, 1, 1, 0])
if let error = error {
    print(error)
}
let time3 = DispatchTime.now()
nn.UpdateWeights()
print("Execution Stats: \((time2.uptimeNanoseconds - time1.uptimeNanoseconds)/1_000_000) ms to make build neural network.")
print("Execution Stats: \((time3.uptimeNanoseconds - time2.uptimeNanoseconds)/1_000_000) ms to calculate error.")
