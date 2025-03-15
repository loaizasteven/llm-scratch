# Self Attention 

The first image demonstrates the concept of Self Attention, which is a mechanism in neural networks that allows the model to focus on different parts of the input sequence when making predictions. This is crucial for understanding the relationships between different parts of the input data.

![Self Attention Image](https://camo.githubusercontent.com/5edb6b2e02db4ad16761a1c6a6de4f75b16d6d03e2eac6d52c263669ea358308/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830335f636f6d707265737365642f31382e77656270)

The second image illustrates Causal Masking, a technique used in sequence models to ensure that the predictions for a given time step depend only on the previous time steps. This is important for maintaining the autoregressive property of models like transformers.

![Causal Masking](https://camo.githubusercontent.com/ae6a1857af914fbb7d57da177ce6bff4b57dbebe3bd11395855b3315ae13d1e1/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f636830335f636f6d707265737365642f31392e77656270)

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

## CausalAttention Module
Similar to `SelfAttention` class, but it includes the ability to mask the future tokens and includes dropout functionality.
