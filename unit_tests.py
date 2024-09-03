#import files
import unittest
import imports as im
import random
import numpy as np

class WholeDangThangTests(unittest.TestCase):
    def getLukesLayers(self):
        return [im.Layer(3, 4, 'sigmoid'),
        im.Layer(4, 8, 'relu'),
        im.Layer(8, 6, 'relu'),
        im.Layer(6, 3, 'relu'),
        im.Layer(3, 1, "linear")]        
    def runNeuralNetwork(self, mapping, layers, trainingSampleCount = 3000, testingSampleCount =10):
        training_samples = {}

        for i in range(trainingSampleCount):
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)
            z = random.uniform(-5,5)
            w = mapping(x,y,z)
            training_samples[(x,y,z)] = w

        testing_samples = {}
        for i in range(testingSampleCount):
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)
            z = random.uniform(-5,5)
            w = mapping(x,y,z)
            testing_samples[(x,y,z)] = w

        network = im.Network(layers, 
                            rate=0.0001,
                            regressor=True)

        return network.train_test(
                        training_samples=training_samples, 
                        testing_samples=testing_samples, 
                        batch_size=64,
                        n_epochs=30
                        )      
    
    def test_whenalwaysretursones_network_should_be_100_percent_right(self):
        def myfunc(x,y,z):
            return 1
        mse = self.runNeuralNetwork(myfunc,self.getLukesLayers())[1]
        self.assertLess(mse, 0.1)
    def test_whenadds_network_should_be_100_percent_right(self):
        def myfunc(x,y,z):
            return x+y+z
        mse = self.runNeuralNetwork(myfunc,self.getLukesLayers())[1]
        self.assertLess(mse, 0.1)        



if __name__ == '__main__':
    unittest.main()