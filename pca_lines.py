# Version 2 with the updated PCA with LINES
import numpy as np
# Import necessary libraries
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained Word2Vec model
model = api.load('glove-wiki-gigaword-50')

def get_word_embeddings(words):
    """ Get vector embeddings for the given list of words. """
    return {word: model[word] for word in words if word in model}

def reduce_dimensions(embeddings):
    """ Reduce dimensions of the embeddings to 2D using PCA. """
    pca = PCA(n_components=2)
    vectors = list(embeddings.values())
    reduced_vectors = pca.fit_transform(vectors)
    return {word: reduced_vectors[i] for i, word in enumerate(embeddings)}

def calculate_angle(point1, point2):
    """ Calculate the polar angle between two points. """
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle = np.arctan2(delta_y, delta_x)
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360

def plot_embeddings_with_lines(embeddings):
    """ Plot embeddings with color-coded lines based on the polar angle. """
    plt.figure(figsize=(10, 10))
    
    # Create a colormap for the angles
    cmap = plt.cm.get_cmap('hsv', 360)

    # Plot each point and draw lines between each pair of points
    for i, (word1, (x1, y1)) in enumerate(embeddings.items()):
        plt.scatter(x1, y1)
        plt.text(x1, y1, word1)
        for word2, (x2, y2) in list(embeddings.items())[i+1:]:
            angle = calculate_angle((x1, y1), (x2, y2))
            plt.plot([x1, x2], [y1, y2], color=cmap(int(angle)))

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Plot of Word Embeddings with Angle-Based Lines')
    plt.grid(True)
    plt.show()

# Example usage:
words = ['king', 'queen', 'man', 'woman', 'apple', 'banana', 'car', 'bike', 'person', 'child', 'toy', 'ice cream', 'food']
embeddings = get_word_embeddings(words)
reduced_embeddings = reduce_dimensions(embeddings)
plot_embeddings_with_lines(reduced_embeddings)
