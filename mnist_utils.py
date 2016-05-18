import numpy as np
import scipy as sc
from NeuralNetwork import *
from NetworkLayer import *
from ActivationAndLossFunctions import *
import numpy.random
import os
from scipy.signal import convolve2d
import scipy.stats
from collections import defaultdict
import scipy.ndimage.filters

def make_snapshots_dir():
    if not os.path.exists('./snapshots'):
        os.makedirs('./snapshots')

# Helpers for contrast_normalize
def l2_norm(image):
    squared = np.square(image*1.0) # Need to cast to floats to avoid integer overflow
    norm_squared = np.sum(squared)
    return math.sqrt(norm_squared)

def l2_normalize(image):
    norm = l2_norm(image)
    if norm == 0:
        return image
    else:
        return image / (norm * 1.0)

def contrast_normalize(reshaped_images):
    """Assumes a reshaped image as input. 
    Normalizes by dividing each pixel value by the l2 norm
    of the pixels in that image."""
    #vector_normalize = np.vectorize(l2_normalize,otypes=[np.float])
    #return vector_normalize(reshaped_images)
    return np.array(map(l2_normalize, reshaped_images))

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    return err_rate[0]

def shuffle_data(X,y):
    """Returns shuffled version of the training data given design matrix X, labels y"""
    n, d = X.shape
    new_indices = np.random.permutation(n)
    shuffled_X = X[new_indices]
    shuffled_y = y[new_indices]
    return shuffled_X,shuffled_y

def split_data(input_features, input_labels, fraction_held_out):
    """Shuffles and splits data into training and test sets.
    Output: training features, training labels, validation features, validation labels"""
    shuffled_features,shuffled_labels = shuffle_data(input_features, input_labels)
    n = input_labels.shape[0]
    cutoff = int(math.floor(n * fraction_held_out))
    training_features = shuffled_features[cutoff:]
    training_labels = shuffled_labels[cutoff:]
    validation_features = shuffled_features[:cutoff]
    validation_labels = shuffled_labels[:cutoff]
    return training_features, training_labels, validation_features, validation_labels

def bag_data(features, labels):
    # Bags the data, sampling with replacement with size n
    num_points = features.shape[0]
    random_indices = numpy.random.choice(num_points, size=num_points, replace=True)
    bagged_features = features[random_indices]
    bagged_labels = labels[random_indices]
    return bagged_features, bagged_labels

# Export CSV
def export_kaggle_csv(labels,filename):
    output = open(filename, "w")
    output.write("Id,Category\n")
    for i in xrange(len(labels)):
        output.write(str(i+1))
        output.write(",")
        output.write(str(labels[i][0]))
        output.write("\n")
    output.close

def generate_new_images(images,labels, num_copies, sigma=6, alpha=36):
    result_images = images
    result_labels = labels
    for i in xrange(num_copies):
        transformed = np.apply_along_axis(lambda x: elastic_transform(x, (28,28), sigma=sigma, alpha=alpha),1,images)
        result_images = np.vstack((result_images, transformed))
        result_labels = np.vstack((result_labels, labels))
    return result_images, result_labels


def elastic_transform(image, square_shape, sigma, alpha, kernel_dim=13):
    """
    This method performs elastic transformations on an image by convolving 
    with a gaussian kernel.
    
    square_shape: dimensions of image to transform into
    kernel_dim: dimension(1-D) of the gaussian kernel
    sigma: standard deviation of the kernel
    alpha: a multiplicative factor for image after convolution
    returns: a nd array transformed image
    """
    orig_shape = image.shape
    image = np.reshape(image, square_shape)

    # create random displacement fields
    displacement_field_x = np.random.uniform(-1,1,size=image.shape) * alpha
    displacement_field_y = np.random.uniform(-1,1,size=image.shape) * alpha
    displacement_field_x = scipy.ndimage.filters.gaussian_filter(displacement_field_x, 6, truncate=1.0)
    displacement_field_y = scipy.ndimage.filters.gaussian_filter(displacement_field_y, 6, truncate=1.0)

    # Normalize to norm 1
    displacement_field_x = l2_normalize(displacement_field_x)
    displacement_field_y = l2_normalize(displacement_field_y)
    # create an empty image
    result = numpy.zeros(square_shape)

    # make the distorted image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields
    for row in xrange(image.shape[1]):
        for col in xrange(image.shape[0]):
            low_ii = row + int(math.floor(displacement_field_x[row, col]))
            high_ii = row + int(math.ceil(displacement_field_x[row, col]))

            low_jj = col + int(math.floor(displacement_field_y[row, col]))
            high_jj = col + int(math.ceil(displacement_field_y[row, col]))

            if (low_ii < 0) or (low_jj < 0) or (high_ii >= image.shape[1]) \
               or (high_jj >= image.shape[0]):
                continue

            res = image[low_ii, low_jj]/4.0 + image[low_ii, high_jj]/4.0 + \
                    image[high_ii, low_jj]/4.0 + image[high_ii, high_jj]/4.0

            result[row, col] = res


    return np.reshape(result, orig_shape)


def get_modes(predictions):
    modes = []
    for i in xrange(predictions.shape[0]):
        counts = defaultdict(int)
        for pred in predictions[i]:
            counts[pred] += 1
        modes.append(max(counts,key=counts.get)) # argmax of the dict
    return modes


def combine_predictions(filenames, output_name):
    """Combine predcitions from list of csvs for an ensemble output."""
    all_predictions = []
    for file in filenames:
        with open(file,'r+') as f:
            predictions = []
            f.readline()
            for line in f:
                prediction = int(line.split(',')[1].rstrip('\n'))
                predictions.append(prediction)
            all_predictions.append(predictions)

    all_predictions = np.array(all_predictions).transpose()
    combined_predictions = get_modes(all_predictions)
    output = open(output_name, "w")
    output.write("Id,Category\n")
    for i in xrange(len(combined_predictions)):
        output.write(str(i+1))
        output.write(",")
        output.write(str(combined_predictions[i]))
        output.write("\n")
    output.close()

