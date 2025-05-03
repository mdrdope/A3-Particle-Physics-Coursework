import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Conv1D, GlobalAveragePooling1D, Masking, Input, Layer
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    A single Transformer Encoder layer:
      1) Multi-Head Attention (MHA)
      2) Residual Connection + Layer Normalization
      3) Feed-Forward Network (FFN)
      4) Residual Connection + Layer Normalization

    Note: This layer receives a mask in call() and passes it to the MultiHeadAttention,
    but does not propagate the mask downstream (i.e., compute_mask returns None).
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_model)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1   = Dropout(dropout_rate)
        self.dropout2   = Dropout(dropout_rate)

    def build(self, input_shape):
        # input_shape is expected to be (batch, seq_length, d_model)
        self.mha.build(input_shape, input_shape, input_shape)
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False, mask=None):
        # Use the passed mask in MultiHeadAttention to ensure that padded positions are masked during attention computation.
        attn_output = self.mha(x, x, x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-Forward Network part; the mask is not needed here.
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

    def compute_mask(self, inputs, mask=None):
        # Do not propagate the mask downstream.
        return None
    

# --------------------------------------------
# 3. Build Transformer (fix dimension mismatch)
# --------------------------------------------

# Custom layer to compute particle mask
@register_keras_serializable()  # @register_keras_serializable(package="MyPackage")
class ParticleMask(Layer):
    def call(self, inputs):
        # For each particle, check if the first column is not equal to -999
        return tf.not_equal(inputs[:, :, 0], -999)

    def compute_output_shape(self, input_shape):
        # Input shape: (batch, max_particles, nParFeatures)
        # Output shape: (batch, max_particles)
        return (input_shape[0], input_shape[1])

@register_keras_serializable()
class VertexMask(Layer):
    def call(self, inputs):
        # Assume all vertices are valid, output a tensor of ones (boolean)
        # return tf.ones(tf.shape(inputs)[:2], dtype=tf.bool)
        return tf.not_equal(inputs[:, :, 0], -999)
    def compute_output_shape(self, input_shape):
        # Input shape: (batch, max_vertices, nVerFeatures)
        # Output shape: (batch, max_vertices)
        return (input_shape[0], input_shape[1])

@register_keras_serializable()
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        # Expand dims along axis=1
        return tf.expand_dims(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        # Input shape: (batch, seq_length)
        # Output shape: (batch, 1, seq_length)
        return (input_shape[0], 1, input_shape[1])

def build_transformer(
    num_layers=4,
    d_model=64,
    num_heads=4,
    d_ff=128,
    dropout_rate=0.1,
    max_particles=35,
    max_vertices=5,
    nParFeatures=14,
    nVerFeatures=8,):
    # Input layers
    inp_particles = Input(shape=(max_particles, nParFeatures), name="particle_input")
    masked_particles = Masking(mask_value=-999)(inp_particles)

    inp_vertices = Input(shape=(max_vertices, nVerFeatures), name="vertex_input")

    # --- Manually construct the mask without Lambda layers ---
    # For particles, compute mask using ParticleMask layer
    particle_mask = ParticleMask()(inp_particles)
    # For vertices, compute mask using VertexMask layer
    vertex_mask = VertexMask()(inp_vertices)
    # Concatenate both masks to obtain a shape of (batch, max_particles + max_vertices)
    combined_mask = Concatenate(axis=1)([particle_mask, vertex_mask])
    # Expand dimensions to obtain a shape of (batch, 1, total_sequence_length)
    combined_mask = ExpandDimsLayer()(combined_mask)
    # --- End mask construction ---

    # Map the inputs to the d_model dimension using Dense layers
    embed_par = Dense(d_model, name="embed_particle")(masked_particles)
    embed_ver = Dense(d_model, name="embed_vertex")(inp_vertices)

    # Concatenate the embeddings from particles and vertices to form a sequence of length (max_particles + max_vertices)
    combined_seq = Concatenate(axis=1, name="concat_particles_vertices")([embed_par, embed_ver])

    # Use the concatenated sequence as input
    x = combined_seq

    # Stack Transformer Encoder layers: pass the same mask to every layer
    for i in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate, name=f"encoder_layer_{i}")(x, mask=combined_mask)

    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name="conv_aggregator")(x)
    x = GlobalAveragePooling1D(name="pooling")(x)

    x = Dense(64, activation='relu', name="dense_head")(x)
    out = Dense(3, activation='softmax', name="flavor_output")(x)

    model = Model(inputs=[inp_particles, inp_vertices], outputs=out, name="TransformerFlavorTagger")
    return model