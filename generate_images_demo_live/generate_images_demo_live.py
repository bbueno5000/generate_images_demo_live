"""
TODO: docstring
"""
import matplotlib.pyplot as pyplot
import numpy
import os
import sys
import tensorflow
import time

class GenerateImages:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        self.mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(
            '/tmp/data/', one_hot=True)
        self.n_pixels = 28 * 28
        self.X = tensorflow.placeholder(tensorflow.float32, shape=([None, self.n_pixels]))
   
    def __call__(self):
        """
        TODO: docstring
        """
        self.__create_model()
        self.__train_model()
        # results
        load_model = False
        if load_model:
            saver.restore(sess, os.path.join(os.getcwd(), 'Trained Bernoulli VAE'))
        num_pairs = 10
        image_indices = numpy.random.randint(0, 200, num_pairs)
        for pair in range(num_pairs):
            x = numpy.reshape(self.mnist.test.images[image_indices[pair]], (1, self.n_pixels))
            pyplot.figure()
            x_image = numpy.reshape(x, (28, 28))
            pyplot.subplot(121)
            pyplot.imshow(x_image)
            x_reconstruction = reconstruction.eval(feed_dict={X: x})
            x_reconstruction_image = (numpy.reshape(x_reconstruction, (28, 28)))
            pyplot.subplot(122)
            pyplot.imshow(x_reconstruction_image)

    def __create_model(self):
        """
        TODO: docstring
        """
        # ENCODER
        latent_dim = 20
        h_dim = 500
        # layer 1
        W_enc = self.weight_variable([self.n_pixels, h_dim], 'W_enc')
        b_enc = self.bias_variable([h_dim], 'b_enc')
        h_enc = tensorflow.nn.tanh(self.FC_layer(self.X, W_enc, b_enc))
        # layer 2
        W_mu = self.weight_variable([h_dim, latent_dim], 'W_mu')
        b_mu = self.bias_variable([latent_dim], 'b_mu')
        mu = self.FC_layer(h_enc, W_mu, b_mu)
        W_logstd = self.weight_variable([h_dim, latent_dim], 'W_logstd')
        b_logstd = self.bias_variable([latent_dim], 'b_logstd')
        logstd = self.FC_layer(h_enc, W_logstd, b_logstd)
        noise = tensorflow.random_normal([1, latent_dim])
        z = mu + tensorflow.mul(noise, tensorflow.exp(.5*logstd))
        # DECODER
        W_dec = self.weight_variable([latent_dim, h_dim], 'W_dec')
        b_dec = self.bias_variable([h_dim], 'b_dec')
        h_dec = tensorflow.nn.tanh(self.FC_layer(z, W_dec, b_dec))
        W_reconstruct = self.weight_variable([h_dim, self.n_pixels], 'W_reconstruct')
        b_reconstruct = self.bias_variable([self.n_pixels], 'b_reconstruct')
        reconstruction = tensorflow.nn.sigmoid(self.FC_layer(h_dec, W_reconstruct, b_reconstruct))
        log_likelihood = tensorflow.reduce_sum(
            self.X * tensorflow.log(reconstruction + 1e-9) + (1 - self.X) * tensorflow.log(
                1 - reconstruction + 1e-9), reduction_indices=1)
        KL_term = -0.5 * tensorflow.reduce_sum(
            1 + 2 * logstd - tensorflow.pow(mu, 2) - tensorflow.exp(2 * logstd), reduction_indices=1)
        variational_lower_bound = tensorflow.reduce_mean(log_likelihood - KL_term)
        optimizer = tensorflow.train.AdadeltaOptimizer().minimize(-variational_lower_bound)

    def __train_model(self):
        """
        TODO: docstring
        """        
        init = tensorflow.global_variables_initializer()
        sess = tensorflow.InteractiveSession()
        sess.run(init)
        saver = tensorflow.train.Saver()
        num_iterations = 1000000
        recording_interval = 1000
        variational_lower_bound_array = list()
        log_likelihood_array = list()
        KL_term_array = list()
        iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]
        for i in range(num_iterations):
            x_batch = numpy.round(self.mnist.train.next_batch(200)[0])
            sess.run(optimizer, feed_dict={X: x_batch})
            if (i%recording_interval == 0):
                vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
                print('Iteration: {}, Loss: {}').format(i, vlb_eval)
                variational_lower_bound_array.append(vlb_eval)
                log_likelihood_array.append(numpy.mean(log_likelihood.eval(feed_dict={X: x_batch})))
                KL_term_array.append(numpy.mean(KL_term.eval(feed_dict={X: x_batch})))
        pyplot.figure()
        pyplot.plot(iteration_array, variational_lower_bound_array)
        pyplot.plot(iteration_array, KL_term_array)
        pyplot.plot(iteration_array, log_likelihood_array)
        pyplot.legend(
            ['Variational Lower Bound', 'KL divergence', 'Log Likelihood'],
            bbox_to_anchor=(1.05, 1), loc=2)
        pyplot.title('Loss per iteration')

    def bias_variable(self, shape, name):
        """
        Bias nodes are added to increase the flexibility of 
        the model to fit the data. Specifically, it allows the 
        network to fit the data when all input features are equal to 00, 
        and very likely decreases the bias of the fitted values elsewhere in the data space
        """
        initial = tensorflow.truncated_normal(shape, stddev=0.1)
        return tensorflow.Variable(initial, name=name)

    def FC_layer(self, X, W, b):
        """
        Neurons in a fully connected layer have full connections to 
        all activations in the previous layer, as seen in regular Neural Networks. 
        Their activations can hence be computed with a matrix multiplication followed by a bias offset.
        """
        return tensorflow.matmul(X, W) + b

    def weight_variable(self, shape, name):
        """
        Represents the strength of connections between units.
        """
        initial = tensorflow.truncated_normal(shape, stddev=0.1)
        return tensorflow.Variable(initial, name=name)

def main(argv):
    """
    TODO: docstring
    """
    generate_images = GenerateImages()
    generate_images()

if __name__ == '__main__':
    main(sys.argv)
