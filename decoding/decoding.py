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


def beam_search(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str], # list of input prompts
    max_tokens: int,
    k: int,
    alpha: float = 0.7,
):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].unsqueeze(1).repeat(1, k, 1)  # (bs, k, seq_len)
        # attention_mask = inputs["attention_mask"].unsqueeze(1).repeat(1, k, 1)

        bs, _, sl = input_ids.size()

        beam_scores = torch.zeros((bs, k), device=input_ids.device)
        beam_scores[:, 1:] = -1e9 # only the 1st beam is considered initially
        finished = torch.zeros((bs, k), dtype=torch.bool, device=input_ids.device)

        # treat each beam as a batch example
        input_ids = input_ids.view(bs*k, sl)
        # attention_mask = attention_mask.view(bs*k, sl)

        for _ in range(max_tokens):
            logits = model(input_ids) # todo: incorporate attn mask and handle unequal prompts in the full logic
            last_logits = logits[:, -1, :] # (bs*k, vocab_size)
            v = last_logits.size(-1)

            log_probs = torch.log_softmax(last_logits, dim=-1)
            log_probs = log_probs.view(bs, k, v)
            candidate_scores = log_probs + beam_scores.unsqueeze(-1) # bs, k, v
            candidate_scores = candidate_scores.view(bs, -1) # bs, k*v

            # (bs, k)
            top_scores, top_idx = torch.topk(candidate_scores, k=k, dim=-1) 
            beam_idx = top_idx // v
            token_idx = top_idx % v

            # get the top beams from prev input_ids and append correct tokens
            input_ids = input_ids.view(bs, k, -1)
            gather_idx = beam_idx.unsqueeze(-1).expand(-1, -1, input_ids.size(-1))
            input_ids = input_ids.gather(1, gather_idx) # (bs, k, seq_len)
            input_ids = torch.cat([input_ids, token_idx.unsqueeze(-1)], dim=-1) # (bs, k, prev_seq_len+1)

            beam_scores = top_scores
            input_ids = input_ids.view(bs*k, -1)

            # todo: check finished beams, store them separately

    return input_ids.view(bs, k, -1), beam_scores
            