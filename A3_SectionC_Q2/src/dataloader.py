import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class HDF5DataLoader(Sequence):
    def __init__(self, h5file, group_name, batch_size=256, shuffle=True):
        self.h5file = h5file
        self.group_name = group_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load the entire HDF5 dataset into memory at once
        with h5py.File(self.h5file, 'r') as f:
            grp = f[self.group_name]
            # Assuming the dataset size is small enough to fit in memory
            self.particles = grp['particles'][:]  # shape (n_samples, 35, 14)
            self.vertices  = grp['vertices'][:]   # shape (n_samples, 5, 8)
            self.labels    = grp['labels'][:]     # shape (n_samples, ...), e.g., one-hot encoding

        self.n_samples = self.particles.shape[0]
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Construct a tf.data.Dataset to speed up data loading and preprocessing
        self.dataset = tf.data.Dataset.from_tensor_slices((
            {
                "particle_input": self.particles,
                "vertex_input": self.vertices
            },
            self.labels))
        if self.shuffle:
            # Use the entire dataset size as buffer_size to ensure thorough shuffling
            self.dataset = self.dataset.shuffle(buffer_size=self.n_samples, reshuffle_each_iteration=True)
        self.dataset = self.dataset.batch(self.batch_size)
        # Use AUTOTUNE for prefetching
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, idx):
        # Obtain a batch by slicing the numpy arrays in memory
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_particles = self.particles[batch_indices]
        X_vertices = self.vertices[batch_indices]
        y = self.labels[batch_indices]
        return {"particle_input": X_particles, "vertex_input": X_vertices}, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_dataset(self):
        """
        Return a tf.data.Dataset object that can be directly passed to model.fit.
        """
        return self.dataset
