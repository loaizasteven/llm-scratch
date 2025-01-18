# Self Attention 

## SelfAttention Module
**Class Initialization**
The __init__ method initializes the SelfAttention class with the following parameters:

- d_in: Dimension of the input vectors.
- d_out: Dimension of the output vectors.
- dqv_bias: Boolean indicating whether to include a bias term in the linear transformations.

**Linear Transformations**
Three linear transformations are defined:

- W_query: Transforms the input vectors into query vectors.
- W_key: Transforms the input vectors into key vectors.
- W_value: Transforms the input vectors into value vectors.
These transformations are represented by nn.Linear layers, which perform a linear mapping from d_in to d_out dimensions.

**Vector Representation**
Input Vector (x): This is the input to the self-attention mechanism. It has a shape of (batch_size, seq_length, d_in).

Query, Key, and Value Vectors:

- Query (q): Obtained by applying W_query to x. Shape: (batch_size, seq_length, d_out).
- Key (k): Obtained by applying W_key to x. Shape: (batch_size, seq_length, d_out).
- Value (v): Obtained by applying W_value to x. Shape: (batch_size, seq_length, d_out).

**Attention Mechanism**
- Attention Scores (attn_score): Calculated by taking the dot product of the query and key vectors. This results in a matrix of shape (batch_size, seq_length, seq_length).

- Attention Weights (attn_weights): Obtained by applying the softmax function to the attention scores, normalized by the square root of the key dimension. This ensures that the weights sum to 1 across the sequence length.

- Context Vector (context_vector): Calculated by taking the dot product of the attention weights and the value vectors. This results in a matrix of shape (batch_size, seq_length, d_out).