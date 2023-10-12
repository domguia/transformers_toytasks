#!/usr/bin/env python

# @XREMOTE_HOST: elk.fleuret.org
# @XREMOTE_EXEC: python
# @XREMOTE_PRE: source ${HOME}/misc/venv/pytorch/bin/activate
# @XREMOTE_PRE: killall -u ${USER} -q -9 python || true
# @XREMOTE_PRE: ln -sf ${HOME}/data/pytorch ./data
# @XREMOTE_SEND: *.py *.sh

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, sys

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################

nb_quantization_levels = 101


def quantize(x, xmin, xmax):
    return (
        ((x - xmin) / (xmax - xmin) * nb_quantization_levels)
        .long()
        .clamp(min=0, max=nb_quantization_levels - 1)
    )


def dequantize(q, xmin, xmax):
    return q / nb_quantization_levels * (xmax - xmin) + xmin


######################################################################


def create_model():
    hidden_dim = 32

    model = nn.Sequential(
        nn.Linear(2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2),
    )

    return model


######################################################################


def generate_sets_and_params(
    nb_mlps,
    nb_samples,
    batch_size,
    nb_epochs,
    device=torch.device("cpu"),
    print_log=False,
):
    data_input = torch.zeros(nb_mlps, 2 * nb_samples, 2, device=device)
    data_targets = torch.zeros(
        nb_mlps, 2 * nb_samples, dtype=torch.int64, device=device
    )

    while (data_targets.float().mean(-1) - 0.5).abs().max() > 0.1:
        i = (data_targets.float().mean(-1) - 0.5).abs() > 0.1
        nb = i.sum()
        print(f"{nb=}")

        nb_rec = 2
        support = torch.rand(nb, nb_rec, 2, 3, device=device) * 2 - 1
        support = support.sort(-1).values
        support = support[:, :, :, torch.tensor([0, 2])].view(nb, nb_rec, 4)

        x = torch.rand(nb, 2 * nb_samples, 2, device=device) * 2 - 1
        y = (
            (
                (x[:, None, :, 0] >= support[:, :, None, 0]).long()
                * (x[:, None, :, 0] <= support[:, :, None, 1]).long()
                * (x[:, None, :, 1] >= support[:, :, None, 2]).long()
                * (x[:, None, :, 1] <= support[:, :, None, 3]).long()
            )
            .max(dim=1)
            .values
        )

        data_input[i], data_targets[i] = x, y

    train_input, train_targets = (
        data_input[:, :nb_samples],
        data_targets[:, :nb_samples],
    )
    test_input, test_targets = data_input[:, nb_samples:], data_targets[:, nb_samples:]

    q_train_input = quantize(train_input, -1, 1)
    train_input = dequantize(q_train_input, -1, 1)
    train_targets = train_targets

    q_test_input = quantize(test_input, -1, 1)
    test_input = dequantize(q_test_input, -1, 1)
    test_targets = test_targets

    hidden_dim = 32
    w1 = torch.randn(nb_mlps, hidden_dim, 2, device=device) / math.sqrt(2)
    b1 = torch.zeros(nb_mlps, hidden_dim, device=device)
    w2 = torch.randn(nb_mlps, 2, hidden_dim, device=device) / math.sqrt(hidden_dim)
    b2 = torch.zeros(nb_mlps, 2, device=device)

    w1.requires_grad_()
    b1.requires_grad_()
    w2.requires_grad_()
    b2.requires_grad_()
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=1e-2)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for k in range(nb_epochs):
        acc_train_loss = 0.0
        nb_train_errors = 0

        for input, targets in zip(
            train_input.split(batch_size, dim=1), train_targets.split(batch_size, dim=1)
        ):
            h = torch.einsum("mij,mnj->mni", w1, input) + b1[:, None, :]
            h = F.relu(h)
            output = torch.einsum("mij,mnj->mni", w2, h) + b2[:, None, :]
            loss = F.cross_entropy(
                output.reshape(-1, output.size(-1)), targets.reshape(-1)
            )
            acc_train_loss += loss.item() * input.size(0)

            wta = output.argmax(-1)
            nb_train_errors += (wta != targets).long().sum(-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for p in [w1, b1, w2, b2]:
                m = (
                    torch.rand(p.size(), device=p.device) <= k / (nb_epochs - 1)
                ).long()
                pq = quantize(p, -2, 2)
                p[...] = (1 - m) * p + m * dequantize(pq, -2, 2)

        train_error = nb_train_errors / train_input.size(1)
        acc_train_loss = acc_train_loss / train_input.size(1)

        # print(f"{k=} {acc_train_loss=} {train_error=}")

    q_params = torch.cat(
        [quantize(p.view(nb_mlps, -1), -2, 2) for p in [w1, b1, w2, b2]], dim=1
    )
    q_train_set = torch.cat([q_train_input, train_targets[:, :, None]], -1).reshape(
        nb_mlps, -1
    )
    q_test_set = torch.cat([q_test_input, test_targets[:, :, None]], -1).reshape(
        nb_mlps, -1
    )

    return q_train_set, q_test_set, q_params


######################################################################


def evaluate_q_params(q_params, q_set, batch_size=25, device=torch.device("cpu")):
    nb_mlps = q_params.size(0)
    hidden_dim = 32
    w1 = torch.empty(nb_mlps, hidden_dim, 2, device=device)
    b1 = torch.empty(nb_mlps, hidden_dim, device=device)
    w2 = torch.empty(nb_mlps, 2, hidden_dim, device=device)
    b2 = torch.empty(nb_mlps, 2, device=device)

    with torch.no_grad():
        k = 0
        for p in [w1, b1, w2, b2]:
            print(f"{p.size()=}")
            x = dequantize(q_params[:, k : k + p.numel() // nb_mlps], -2, 2).view(
                p.size()
            )
            p.copy_(x)
            k += p.numel() // nb_mlps

    q_set = q_set.view(nb_mlps, -1, 3)
    data_input = dequantize(q_set[:, :, :2], -1, 1).to(device)
    data_targets = q_set[:, :, 2].to(device)

    print(f"{data_input.size()=} {data_targets.size()=}")

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    acc_loss = 0.0
    nb_errors = 0

    for input, targets in zip(
        data_input.split(batch_size, dim=1), data_targets.split(batch_size, dim=1)
    ):
        h = torch.einsum("mij,mnj->mni", w1, input) + b1[:, None, :]
        h = F.relu(h)
        output = torch.einsum("mij,mnj->mni", w2, h) + b2[:, None, :]
        loss = F.cross_entropy(output.reshape(-1, output.size(-1)), targets.reshape(-1))
        acc_loss += loss.item() * input.size(0)
        wta = output.argmax(-1)
        nb_errors += (wta != targets).long().sum(-1)

    error = nb_errors / data_input.size(1)
    acc_loss = acc_loss / data_input.size(1)

    return error


######################################################################


def generate_sequence_and_test_set(
    nb_mlps,
    nb_samples,
    batch_size,
    nb_epochs,
    device,
):
    q_train_set, q_test_set, q_params = generate_sets_and_params(
        nb_mlps,
        nb_samples,
        batch_size,
        nb_epochs,
        device=device,
    )

    input = torch.cat(
        [
            q_train_set,
            q_train_set.new_full(
                (
                    q_train_set.size(0),
                    1,
                ),
                nb_quantization_levels,
            ),
            q_params,
        ],
        dim=-1,
    )

    print(f"SANITY #1 {q_train_set.size()=} {q_params.size()=} {input.size()=}")

    ar_mask = (
        (torch.arange(input.size(0), device=input.device) > q_train_set.size(0) + 1)
        .long()
        .view(1, -1)
        .reshape(nb_mlps, -1)
    )

    return input, ar_mask, q_test_set


######################################################################

if __name__ == "__main__":
    import time

    nb_mlps, nb_samples = 128, 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()

    data = []

    for n in range(2):
        data.append(
            generate_sequence_and_test_set(
                nb_mlps=nb_mlps,
                nb_samples=nb_samples,
                device=device,
                batch_size=25,
                nb_epochs=250,
            )
        )

    end_time = time.perf_counter()
    nb = sum([i.size(0) for i, _, _ in data])
    print(f"{nb / (end_time - start_time):.02f} samples per second")

    for input, ar_mask, q_test_set in data:
        q_train_set = input[:, : nb_samples * 3]
        q_params = input[:, nb_samples * 3 + 1 :]
        print(f"SANITY #2 {q_train_set.size()=} {q_params.size()=} {input.size()=}")
        error_train = evaluate_q_params(q_params, q_train_set)
        print(f"train {error_train*100}%")
        error_test = evaluate_q_params(q_params, q_test_set)
        print(f"test {error_test*100}%")