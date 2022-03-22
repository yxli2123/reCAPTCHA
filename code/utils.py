import torch
import numpy as np
import random
import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(y_gt, y_pr, tokenizer, num_character=1, top_k=1):
    """
    :param          y_gt: of shape (N*num_char, vocab_size)
    :param          y_pr: of shape (N*num_char, 1)
    :param     tokenizer: for decode
    :param num_character:
    :param         top_k: top k candidates of the prediction
    :return:
    """

    # Statics
    acc_single = 0
    acc_pair = 0
    acc_topk = 0
    loss = F.cross_entropy(y_pr, y_gt.squeeze())

    # Results
    results = []

    N_num_char, vocab_size = y_pr.shape
    y_pr = y_pr.view(N_num_char // num_character, num_character, vocab_size)  # (N, num_char, vocab_size)
    y_gt = y_gt.view(N_num_char // num_character, num_character)              # (N, num_char)
    N, _, _ = y_pr.shape

    for gt, pr in zip(y_gt, y_pr):
        """
        pr of shape (num_char, vocab_size)
        gt of shape (num_char)
        """
        top_k_value, top_k_token = torch.topk(pr, k=top_k, dim=1)

        # Statics
        correct_cnt = 0
        for i in range(num_character):
            # Single character accuracy
            if top_k_token[i][0] == gt[i]:
                correct_cnt += 1
                acc_single += 1

            # Top k character accuracy: if ground truth is among the top k answer
            for j in range(top_k):
                if top_k_token[i][j] == y_gt[i]:
                    acc_topk += 1

            if correct_cnt == num_character:
                acc_pair += 1

        # Results
        char_gt = tokenizer.decode(gt)  # e.g. "中国"
        char_pr = " ".join([tokenizer.decode(tokens) for tokens in top_k_token])  # e.g. "中口申 国图固"
        results.append({"char_gt": char_gt,
                        "char_pr": char_pr})

    metric = {"loss": loss,
              "acc_single": acc_single / (num_character * N),
              "acc_pair": acc_pair / N,
              "acc_topk": acc_topk / (num_character * N),
              "results": results}
    return metric
