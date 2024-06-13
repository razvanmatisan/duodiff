from CMMD_evaluation import distance, embedding, io_util
import numpy as np


def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError(
            "`ref_dir` and `ref_embed_file` both cannot be set at the same time."
        )
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(
            ref_dir, embedding_model, batch_size, max_count
        ).astype("float32")
    eval_embs = io_util.compute_embeddings_for_dir(
        eval_dir, embedding_model, batch_size, max_count
    ).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()
