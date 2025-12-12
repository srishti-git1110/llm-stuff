from typing import Optional
import torch

def greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    greedy decoding
    
    :param logits: logits tensor (bs, seq_len, vocab_size)
    :type logits: torch.Tensor
    :return: next token ids for each seq (bs, 1)
    :rtype: torch.Tensor
    """

    last_logits = logits[:, -1, :]
    return torch.argmax(last_logits, dim=-1, keepdim=False)


def top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    top-k sampling decoding
    
    :param logits: logits tensor (bs, vocab_size)
    :param k: top k value

    :return: next token ids for each seq (bs, 1)
    :rtype: torch.Tensor
    """
    topk, topk_idx = torch.topk(logits, k=k, dim=-1) # (bs, k)
    topk_probs = torch.softmax(topk, dim=-1)
    sample = torch.multinomial(topk_probs, num_samples=1)
    return topk_idx.gather(1, sample)


def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    top-p (nucleus) sampling decoding
    
    :param logits: logits tensor (bs, vocab_size)
    :param p: top p value
    :return: next token ids for each seq (bs, 1)
    :rtype: torch.Tensor
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    mask = cum_probs <= p
    mask[..., 0] = True # in case prob for token 1 is already > p
    masked_logits = torch.where(mask, sorted_logits, torch.tensor(float('-inf')))

    new_probs = torch.softmax(masked_logits, dim=-1)
    sample = torch.multinomial(new_probs, num_samples=1) # (bs, 1)
    return sorted_idx.gather(1, sample).squeeze(-1)


def temp_top_p_top_k(logits: torch.Tensor, p: Optional[float]=None, k: Optional[int]=None, temperature: float = 1.0, ) -> torch.Tensor:
    """
    temperature scaling decoding
    
    :param logits: logits tensor (bs, seq_len, vocab_size)
    :param temperature: temperature value
    :param p: top p value
    :param k: top k value
    :return: next token ids for each seq (bs, )
    :rtype: torch.Tensor
    """
    if p is not None and k is not None:
        raise ValueError("Only one of p or k should be set.") 

    last_logits = logits[:, -1, :]
    scaled_logits = last_logits / temperature

    if k is not None:
        if k < 1:
            raise ValueError("k must be greater than 0.")
        return top_k(scaled_logits, k)
    
    if p is not None:
        if p <= 0.0 or p > 1.0:
            raise ValueError("p must be in the range (0.0, 1.0].")
        return top_p(scaled_logits, p)