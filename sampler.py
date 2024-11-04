import torch
import numpy as np

class Sampler:

    def __init__(
        self,
        top_k=None,
        top_p=None,
        frequency_penalty=1.0,
        presence_penalty=1.0
    ):
        '''
        param top_k : (None or int)
            If specified, only the top k logits should be used during sampling
            If this is specified, top_p should be None

        param top_p : (None or int)
            If specified, only the logits representing the probability mass p should be used during sampling.
            Or, if the top token has mass greater than p, the top token is returned.
            If this is specified, top_k should be None

        If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

        param frequency_penalty : (float)
            A penalty applied to tokens that have previously occured in the sequence. Along with
            presence_penalty, this adjusts the per-token softmax temperature.
            A penalty of 1.0 indicates no change from normal softmax.

        param presence_penalty : (float)
            A penalty applied to tokens IF they have previously occured in the sequence. Along with
            frequency_penalty, this adjusts the per-token softmax temperature.
            A penalty of 1.0 indicates no change from normal softmax.
        '''
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.fp_incr = frequency_penalty - 1.0
        self.pp_incr = presence_penalty - 1.0
        
        if top_k is None and top_p is None:
            self.top_k = None
            self.top_p = 1.0
        elif top_k is not None:
            self.top_k = top_k
            self.top_p = None
        else:
            self.top_k = None
            self.top_p = top_p


    def sample_token(self, raw_unsorted_logits, previous_token_ids):
        '''
        param: raw_unsorted_logits (float numpy array)
            A one dimensional list of logits representing an unnormalized distribution over next tokens

        param: previous_token_ids (int numpy array)
            A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

        returns: a single token id (integer), sampled according to the specified sampling parameters
        '''

        logits = raw_unsorted_logits

        # for the solution will just assume top-p and frequency=1.1
        temperatures = np.ones_like(raw_unsorted_logits)

        # add frequency penalty
        for tid in previous_token_ids:
            temperatures[tid] += self.fp_incr

        # add presence penalty
        uniqs = set()
        for tid in previous_token_ids:
            uniqs.add(tid)
        for tid in uniqs:
            temperatures[tid] += self.pp_incr

        # bump to positive logits
        logits = logits - np.min(logits)

        # apply temperatures
        logits = (logits / temperatures)

        # max trick, idk if this matters.
        logits = logits - np.max(logits)

        # apply softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # sort in descending order
        indices = np.argsort(probs)
        probs = np.sort(probs)
        indices = indices[::-1]
        probs = probs[::-1]

        # compute cutoffs
        if self.top_k is not None:
            cutoff = self.top_k
        else:
            cumsum = 0.0
            cutoff = 0
            for i in range(len(probs)):
                cutoff += 1
                cumsum += probs[i]
                if cumsum >= self.top_p:
                    break

        indices = indices[:cutoff+1]
        probs = probs[:cutoff+1]

        # renormalize
        probs = (probs / np.sum(probs))

        # sample
        return np.random.choice(indices, p=probs)


    # an alternative way to call sample_token(), for convenience
    def __call__(self, raw_unsorted_logits, previous_token_ids):
        return self.sample_token(raw_unsorted_logits, previous_token_ids)
