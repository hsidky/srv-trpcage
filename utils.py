def chunk_data(data, n_chunks):
    n_data = data.shape[0]
    batch_size = n_data//n_chunks
    chunked_data = [data[i:i + batch_size] for i in range(0, n_data, batch_size)]
    return chunked_data