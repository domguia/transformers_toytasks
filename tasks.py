#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math, os, tqdm

import torch, torchvision

from torch import nn
from torch.nn import functional as F

from mygpt import BracketedSequence

# from graph import save_attention_image
save_attention_image = None

######################################################################


def masked_inplace_autoregression(
    model,
    batch_size,
    input,
    ar_mask,
    deterministic_synthesis,
    forbidden_tokens=None,
    progress_bar_desc="autoregression",
    device=torch.device("cpu"),
):
    assert input.size() == ar_mask.size()

    batches = zip(input.split(batch_size), ar_mask.split(batch_size))

    if progress_bar_desc is not None:
        batches = tqdm.tqdm(
            batches,
            dynamic_ncols=True,
            desc=progress_bar_desc,
            total=(input.size(0) + batch_size - 1) // batch_size,
        )

    with torch.autograd.no_grad():
        t = model.training
        model.eval()

        for input, ar_mask in batches:
            model.masked_inplace_autoregression(
                input, ar_mask, forbidden_tokens, deterministic_synthesis
            )

        model.train(t)


######################################################################


class Task:
    def batches(self, split="train"):
        pass

    def vocabulary_size(self):
        pass

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        pass


class TaskFromFile(Task):
    def tensorize(self, pairs):
        len_max = max([len(x[0]) for x in pairs])

        input = torch.cat(
            [
                torch.tensor(
                    [
                        [self.char2id[c] for c in s[0] + "#" * (len_max - len(s[0]))]
                        for s in pairs
                    ]
                )
            ],
            0,
        ).to("cpu")

        pred_mask = torch.cat(
            [
                torch.tensor(
                    [
                        [int(c) for c in s[1] + "0" * (len_max - len(s[1]))]
                        for s in pairs
                    ]
                )
            ],
            0,
        ).to("cpu")

        return input, pred_mask

    # trim all the tensors in the tuple z to remove as much token from
    # left and right in the first tensor. If z is a tuple, all its
    # elements are trimed according to the triming for the first
    def trim(self, z, token="#"):
        n = self.char2id[token]
        if type(z) == tuple:
            x = z[0]
            i = (1 - (F.pad(x, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return tuple([t[:, a:b] for t in z])
        else:
            i = (1 - (F.pad(z, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return z[:, a:b]

    def __init__(
        self,
        filename,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        device=torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.device = device

        pairs = []
        with open(filename, "r") as f:
            for _ in range(nb_train_samples + nb_test_samples):
                sequence = f.readline().strip()
                pred_mask = f.readline().strip()
                assert len(sequence) == len(pred_mask)
                assert set(pred_mask) == {"0", "1", "2"}, f"{set(pred_mask)}"
                pairs.append((sequence, pred_mask))

        symbols = ["#"] + list(set("".join([x[0] for x in pairs])) - set(["#"]))
        print("SANITY", symbols)
        self.char2id = dict([(c, n) for n, c in enumerate(symbols)])
        self.id2char = dict([(n, c) for c, n in self.char2id.items()])

        print(self.char2id)

        self.train_input, self.train_pred_masks = self.tensorize(
            pairs[:nb_train_samples]
        )
        self.test_input, self.test_pred_masks = self.tensorize(pairs[nb_train_samples:])

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield self.trim(batch).to(self.device)

    def vocabulary_size(self):
        return len(self.char2id)

    def tensor2str(self, t):
        print(f"{type(t)=}")
        return ["".join([self.id2char[x.item()] for x in s]) for s in t]

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        correct = self.trim(self.test_input[:1000]).to(self.device)
        result = correct.clone()
        pred_mask = self.test_pred_masks[:1000, : result.size(1)].to(self.device)
        ar_mask = (pred_mask > 0).long()
        result *= 1 - ar_mask  # paraaaaanoiaaaaaaa

        logger(f"----------------------------------------------------------")

        for e in self.tensor2str(result[:10]):
            logger(f"test_before {e}")

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        logger(f"----------------------------------------------------------")

        for e, c in zip(self.tensor2str(result[:10]), self.tensor2str(correct[:10])):
            logger(f"test_after  {e}")
            logger(f"correct     {c}")

        logger(f"----------------------------------------------------------")

        err_mask = (pred_mask == 2).long()
        nb_total = err_mask.sum().item()
        nb_correct = ((correct == result).long() * err_mask).sum().item()

        logger(f"test_performance {n_epoch} {nb_total=} {nb_correct=}")
        logger(f"main_test_accuracy {n_epoch} {nb_correct / nb_total}")


####################

import problems


class SandBox(Task):
    def __init__(
        self,
        problem,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        logger=None,
        device=torch.device("cpu"),
        max_nb_codes=1024,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.problem = problem

        self.train_input, self.train_ar_mask = self.problem.generate_sequences(
            nb_train_samples
        )
        self.test_input, self.test_ar_mask = self.problem.generate_sequences(
            nb_test_samples
        )

        self.train_input, self.train_ar_mask = self.train_input.to(
            device
        ), self.train_ar_mask.to(device)
        self.test_input, self.test_ar_mask = self.test_input.to(
            device
        ), self.test_ar_mask.to(device)

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

        # A bit of paranoia never hurts
        assert self.nb_codes <= max_nb_codes
        assert self.train_input.min() >= 0
        assert self.test_input.min() >= 0
        assert tuple(x.item() for x in self.train_ar_mask.unique()) in {
            (0,),
            (1,),
            (0, 1),
        }
        assert tuple(x.item() for x in self.test_ar_mask.unique()) in {
            (0,),
            (1,),
            (0, 1),
        }

        if logger is not None:
            for s, a in zip(self.train_input[:100], self.train_ar_mask[:100]):
                logger(f"train_sequences {self.problem.seq2str(s)}")
                a = "".join(["01"[x.item()] for x in a])
                logger(f"                {a}")

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis, nmax=1000
    ):
        def compute_accuracy(input, ar_mask, logger=None):
            input, ar_mask = input[:nmax], ar_mask[:nmax]
            result = input.clone() * (1 - ar_mask)

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )

            log_ground_truth = ar_mask.min() == 0

            if logger is not None:
                for sp, st in zip(result[:10], input[:10]):
                    logger(
                        f"test_sequences {n_epoch} prediction   {self.problem.seq2str(sp)}"
                    )
                    if log_ground_truth:
                        logger(
                            f"               {n_epoch} ground truth {self.problem.seq2str(st)}"
                        )

            nb_total, nb_correct = self.problem.compute_nb_correct(
                input, ar_mask, result
            )

            # nb_total = ar_mask.sum().item()
            # nb_correct = ((result == input).long() * ar_mask).sum().item()

            return nb_total, nb_correct

        train_nb_total, train_nb_correct = compute_accuracy(
            self.train_input, self.train_ar_mask
        )

        logger(
            f"accuracy_train {n_epoch} nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
        )

        test_nb_total, test_nb_correct = compute_accuracy(
            self.test_input, self.test_ar_mask, logger
        )

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )

        logger(f"main_test_accuracy {n_epoch} {test_nb_correct/test_nb_total}")

        if save_attention_image is not None:
            for k in range(10):
                ns = torch.randint(self.test_input.size(0), (1,)).item()
                input = self.test_input[ns : ns + 1].clone()

                with torch.autograd.no_grad():
                    t = model.training
                    model.eval()
                    # model.record_attention(True)
                    model(BracketedSequence(input))
                    model.train(t)
                    # ram = model.retrieve_attention()
                    # model.record_attention(False)

                # tokens_output = [c for c in self.problem.seq2str(input[0])]
                # tokens_input = ["n/a"] + tokens_output[:-1]
                # for n_head in range(ram[0].size(1)):
                # filename = os.path.join(
                # result_dir, f"sandbox_attention_{k}_h{n_head}.pdf"
                # )
                # attention_matrices = [m[0, n_head] for m in ram]
                # save_attention_image(
                # filename,
                # tokens_input,
                # tokens_output,
                # attention_matrices,
                # k_top=10,
                ##min_total_attention=0.9,
                # token_gap=12,
                # layer_gap=50,
                # )
                # logger(f"wrote {filename}")


######################################################################

import picoclvr


class PicoCLVR(Task):
    # Make a tensor from a list of strings
    def tensorize(self, descr):
        token_descr = [s.strip().split(" ") for s in descr]
        l = max([len(s) for s in token_descr])
        token_descr = [s + ["<nul>"] * (l - len(s)) for s in token_descr]
        id_descr = [[self.token2id[u] for u in s] for s in token_descr]
        return torch.tensor(id_descr, device=self.device)

    # Make a list of strings from a tensor
    def detensorize(self, x):
        return [" ".join([self.id2token[t.item()] for t in r]) for r in x]

    # trim all the tensors in the tuple z to remove as much token from
    # left and right in the first tensor. If z is a tuple, all its
    # elements are trimed according to the triming for the first
    def trim(self, z, token="<nul>"):
        n = self.token2id[token]
        if type(z) == tuple:
            x = z[0]
            i = (1 - (F.pad(x, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return tuple([t[:, a:b] for t in z])
        else:
            i = (1 - (F.pad(z, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return z[:, a:b]

    ######################

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_colors=5,
        logger=None,
        device=torch.device("cpu"),
        pruner_train=None,
        pruner_eval=None,
    ):
        super().__init__()

        def generate_descr(nb, cache_suffix, pruner):
            return picoclvr.generate(
                nb,
                height=self.height,
                width=self.width,
                nb_colors=nb_colors,
                pruner=pruner,
            )

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device = device
        self.pruner_train = pruner_train
        self.pruner_eval = pruner_eval

        if logger is not None:
            logger(
                f"generating {nb_train_samples+nb_test_samples} samples (can take some time)"
            )

        self.train_descr = generate_descr(
            nb_train_samples, "train", pruner=self.pruner_train
        )
        self.test_descr = generate_descr(nb_test_samples, "test", pruner=None)

        # Build the tokenizer
        tokens = {"<nul>", "<img>"}
        for d in [self.train_descr, self.test_descr]:
            for s in d:
                for t in s.strip().split(" "):
                    tokens.add(t)
        # make this set a sorted list to get the same tensors given
        # the same descr
        tokens = list(tokens)
        tokens.sort()
        self.token2id = dict([(t, n) for n, t in enumerate(tokens)])
        self.id2token = dict([(n, t) for n, t in enumerate(tokens)])
        self.t_img, self.t_nul = self.token2id["<img>"], self.token2id["<nul>"]

        # Tokenize the train and test sets
        self.train_input = self.tensorize(self.train_descr)
        self.test_input = self.tensorize(self.test_descr)

    def batches(self, split="train"):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=f"epoch-{split}"
        ):
            yield self.trim(batch)

    def vocabulary_size(self):
        return len(self.token2id)

    def compute_missing_properties(
        self, n_epoch, model, logger, deterministic_synthesis, pruner=None
    ):
        acc_nb_requested_properties = []
        acc_nb_missing_properties = []
        acc_nb_results = 0

        for input in tqdm.tqdm(
            self.test_input.split(self.batch_size),
            dynamic_ncols=True,
            desc=f"test-properties",
        ):
            result = input.clone()
            ar_mask = (result == self.t_img).long().cumsum(dim=1).clamp(max=1)
            result = (1 - ar_mask) * result + ar_mask * self.t_nul
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )

            result_descr = self.detensorize(result)
            np = picoclvr.nb_properties(
                result_descr,
                height=self.height,
                width=self.width,
                pruner=pruner,
            )
            nb_requested_properties, _, nb_missing_properties = zip(*np)
            acc_nb_requested_properties += nb_requested_properties
            acc_nb_missing_properties += nb_missing_properties
            acc_nb_results += len(result_descr)

        nb_requested_properties = sum(acc_nb_requested_properties)
        nb_missing_properties = sum(acc_nb_missing_properties)

        prefix = "" if pruner is None else "pruned_"
        logger(f"nb_{prefix}samples {n_epoch} {acc_nb_results}")
        logger(
            f"property_{prefix}nb {n_epoch} requested {sum(acc_nb_requested_properties)} missing {sum(acc_nb_missing_properties)}"
        )
        logger(
            f"property_{prefix}miss {n_epoch} {100*nb_missing_properties/nb_requested_properties:.02f}%"
        )

        logger(
            f"main_test_accuracy {n_epoch} {1-nb_missing_properties/nb_requested_properties}"
        )

    ######################################################################

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        self.compute_missing_properties(n_epoch, model, logger, deterministic_synthesis)

        if self.pruner_eval is not None:
            self.compute_missing_properties(n_epoch, model, self.pruner_eval)

        nb_tokens_to_generate = self.height * self.width + 3
        result_descr = []
        nb_per_primer = 8
        primer = []

        for primer_descr in [
            "red above green <sep> green top <sep> blue right of red",
            "there is red <sep> there is yellow <sep> there is blue",
            "red below yellow <sep> yellow below green <sep> green below blue <sep> red right <sep> yellow left <sep> green right <sep> blue left",
            "green bottom <sep> yellow bottom <sep> green left of blue <sep> yellow right of blue <sep> blue top",
        ]:
            primer += [primer_descr + " <img>"] * nb_per_primer

        result = self.tensorize(primer)
        fill = result.new_full(
            result.size()[:-1] + (self.height * self.width + 1,), self.t_nul
        )
        result = torch.cat((result, fill), 1)
        ar_mask = (result == self.t_nul).long()
        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )
        result_descr = self.detensorize(result)

        np = picoclvr.nb_properties(result_descr, height=self.height, width=self.width)

        acc_nb_requested_properties, _, acc_nb_missing_properties = zip(*np)
        acc_nb_results = len(result_descr)

        nb_requested_properties = sum(acc_nb_requested_properties)
        nb_missing_properties = sum(acc_nb_missing_properties)

        prefix = "demo_"
        logger(f"nb_{prefix}samples {n_epoch} {acc_nb_results}")
        logger(
            f"property_{prefix}nb {n_epoch} requested {sum(acc_nb_requested_properties)} missing {sum(acc_nb_missing_properties)}"
        )
        logger(
            f"property_{prefix}miss {n_epoch} {100*nb_missing_properties/nb_requested_properties:.02f}%"
        )

        img = picoclvr.descr2img(result_descr, height=self.height, width=self.width)

        if img.dim() == 5:
            if img.size(1) == 1:
                img = F.pad(img.squeeze(1), pad=(1, 1, 1, 1), value=64)
            else:
                img = torch.cat(
                    [
                        torchvision.utils.make_grid(x, padding=1, pad_value=64)[None]
                        for x in img
                    ],
                    0,
                )

        image_name = os.path.join(result_dir, f"picoclvr_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            img / 255.0, image_name, nrow=nb_per_primer, padding=1, pad_value=0.0
        )
        logger(f"wrote {image_name}")


######################################################################


class MNIST(Task):
    def __init__(
        self, nb_train_samples, nb_test_samples, batch_size, device=torch.device("cpu")
    ):
        super().__init__()

        self.nb_train_samples = (nb_train_samples,)
        self.nb_test_samples = (nb_test_samples,)
        self.batch_size = batch_size
        self.device = device
        data_set = torchvision.datasets.MNIST(root="./data", train=True, download=True)
        self.train_input = data_set.data[:nb_train_samples].view(-1, 28 * 28).long()
        data_set = torchvision.datasets.MNIST(root="./data", train=False, download=True)
        self.test_input = data_set.data[:nb_test_samples].view(-1, 28 * 28).long()

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return 256

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        results = torch.empty(64, 28 * 28, device=self.device, dtype=torch.int64)
        ar_mask = torch.full_like(results, 1)
        masked_inplace_autoregression(
            model,
            self.batch_size,
            results,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )
        image_name = os.path.join(result_dir, f"mnist_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            1 - results.reshape(-1, 1, 28, 28) / 255.0,
            image_name,
            nrow=16,
            pad_value=0.8,
        )
        logger(f"wrote {image_name}")


######################################################################

import maze


class Maze(Task):
    def map2seq(self, *m):
        return torch.cat([x.flatten(1) for x in m], 1)

    def seq2map(self, s):
        s = s.reshape(s.size(0), -1, self.height, self.width)
        return (s[:, k] for k in range(s.size(1)))

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_walls,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device = device

        train_mazes, train_paths, _ = maze.create_maze_data(
            nb_train_samples,
            height=height,
            width=width,
            nb_walls=nb_walls,
            progress_bar=lambda x: tqdm.tqdm(x, dynamic_ncols=True, desc=f"data-train"),
        )
        self.train_input = self.map2seq(train_mazes.to(device), train_paths.to(device))

        test_mazes, test_paths, _ = maze.create_maze_data(
            nb_test_samples,
            height=height,
            width=width,
            nb_walls=nb_walls,
            progress_bar=lambda x: tqdm.tqdm(x, dynamic_ncols=True, desc=f"data-test"),
        )
        self.test_input = self.map2seq(test_mazes.to(device), test_paths.to(device))

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def compute_error(
        self, model, split="train", nb_to_use=-1, deterministic_synthesis=False
    ):
        nb_total, nb_correct = 0, 0
        count = torch.zeros(
            self.width * self.height,
            self.width * self.height,
            device=self.device,
            dtype=torch.int64,
        )

        for input in self.batches(split, nb_to_use):
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            mazes, paths = self.seq2map(result)
            path_correctness = maze.path_correctness(mazes, paths)
            nb_correct += path_correctness.long().sum()
            nb_total += mazes.size(0)

            optimal_path_lengths = (
                (input[:, self.height * self.width :] == maze.v_path).long().sum(1)
            )
            predicted_path_lengths = (
                (result[:, self.height * self.width :] == maze.v_path).long().sum(1)
            )
            optimal_path_lengths = optimal_path_lengths[path_correctness]
            predicted_path_lengths = predicted_path_lengths[path_correctness]
            count[optimal_path_lengths, predicted_path_lengths] += 1

        if count.max() == 0:
            count = None
        else:
            count = count[
                : count.sum(1).nonzero().max() + 1, : count.sum(0).nonzero().max() + 1
            ]

        return nb_total, nb_correct, count

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        train_nb_total, train_nb_correct, count = self.compute_error(
            model,
            "train",
            nb_to_use=1000,
            deterministic_synthesis=deterministic_synthesis,
        )
        logger(
            f"accuracy_train {n_epoch} nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
        )

        test_nb_total, test_nb_correct, count = self.compute_error(
            model,
            "test",
            nb_to_use=1000,
            deterministic_synthesis=deterministic_synthesis,
        )
        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )

        logger(f"main_test_accuracy {n_epoch} {test_nb_correct/test_nb_total}")

        if count is not None:
            proportion_optimal = count.diagonal().sum().float() / count.sum()
            logger(f"proportion_optimal_test {proportion_optimal*100:.02f}%")
            with open(
                os.path.join(result_dir, f"maze_result_{n_epoch:04d}.txt"), "w"
            ) as f:
                for i in range(count.size(0)):
                    for j in range(count.size(1)):
                        eol = " " if j < count.size(1) - 1 else "\n"
                        f.write(f"{count[i,j]}{eol}")

        input = self.test_input[:48]
        result = input.clone()
        ar_mask = result.new_zeros(result.size())
        ar_mask[:, self.height * self.width :] = 1
        result *= 1 - ar_mask
        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        mazes, paths = self.seq2map(input)
        _, predicted_paths = self.seq2map(result)

        filename = os.path.join(result_dir, f"maze_result_{n_epoch:04d}.png")
        maze.save_image(
            filename,
            mazes=mazes,
            target_paths=paths,
            predicted_paths=predicted_paths,
            path_correct=maze.path_correctness(mazes, predicted_paths),
            path_optimal=maze.path_optimality(paths, predicted_paths),
        )
        logger(f"wrote {filename}")


######################################################################


import snake


class Snake(Task):
    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_colors,
        length,
        prompt_length,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device = device
        self.prompt_length = prompt_length

        self.train_input, self.train_prior_visits, _, _ = snake.generate_sequences(
            nb_train_samples,
            height,
            width,
            nb_colors,
            length,
            prompt_length,
            self.device,
        )
        self.test_input, self.test_prior_visits, _, _ = snake.generate_sequences(
            nb_test_samples,
            height,
            width,
            nb_colors,
            length,
            prompt_length,
            self.device,
        )

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        def compute_nb_correct(input, prior_visits):
            result = input.clone()
            i = torch.arange(result.size(1), device=result.device)[None, :]
            ar_mask = (
                torch.logical_and(i >= self.prompt_length * 2, i % 2 == 0)
                .long()
                .expand_as(result)
            )
            result *= 1 - ar_mask

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            nb_total = ((prior_visits > 0) * ar_mask).sum()

            nb_correct = ((result == input).long() * (prior_visits > 0) * ar_mask).sum()

            return nb_total, nb_correct

        test_nb_total, test_nb_correct = compute_nb_correct(
            self.test_input[:1000], self.test_prior_visits[:1000]
        )

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )

        logger(f"main_test_accuracy {n_epoch} {test_nb_correct/test_nb_total}")


######################################################################


import stack


class Stack(Task):
    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        logger,
        nb_steps,
        nb_stacks,
        nb_digits,
        fraction_values_for_train=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_stacks = nb_stacks
        self.nb_digits = nb_digits
        self.device = device

        if fraction_values_for_train is None:
            values_for_train = None
            values_for_test = None
        else:
            all = torch.randperm(10**nb_digits)
            nb_for_train = int(all.size(0) * fraction_values_for_train)
            values_for_train = all[:nb_for_train]
            values_for_test = all[nb_for_train:]

        self.train_input, self.train_stack_counts = stack.generate_sequences(
            nb_train_samples,
            nb_steps,
            nb_stacks,
            nb_digits,
            values_for_train,
            self.device,
        )

        self.test_input, self.test_stack_counts = stack.generate_sequences(
            nb_test_samples,
            nb_steps,
            nb_stacks,
            nb_digits,
            values_for_test,
            self.device,
        )

        i = torch.logical_and(self.test_input % 2 == 1, self.test_input < 2 * nb_stacks)
        counts = self.test_stack_counts.flatten()[i.flatten()]
        counts = F.one_hot(counts).sum(0)
        logger(f"test_pop_stack_counts {counts}")

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        def compute_nb_correct(input):
            result = input.clone()
            stack.remove_popped_values(result, self.nb_stacks, self.nb_digits)
            ar_mask = (result != input).long()
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            errors = ((result != input).long() * ar_mask).reshape(
                -1, 1 + self.nb_digits
            )
            ar_mask = ar_mask.reshape(-1, 1 + self.nb_digits)

            nb_total = ar_mask.max(1).values.sum()
            nb_correct = nb_total - errors.max(1).values.sum()

            return nb_total, nb_correct

        test_nb_total, test_nb_correct = compute_nb_correct(self.test_input[:1000])

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )

        logger(f"main_test_accuracy {n_epoch} {test_nb_correct/test_nb_total}")

        ##############################################################
        # Log a few generated sequences
        input = self.test_input[:10, : 12 * (1 + self.nb_digits)]
        result = input.clone()
        stack.remove_popped_values(result, self.nb_stacks, self.nb_digits)
        ar_mask = (result != input).long()

        # for n in range(result.size(0)):
        # logger(
        # f"test_before {stack.seq_to_str(result[n],nb_stacks=self.nb_stacks,nb_digits=self.nb_digits)}"
        # )

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        for n in range(result.size(0)):
            logger(
                f"test_after  {stack.seq_to_str(result[n],nb_stacks=self.nb_stacks,nb_digits=self.nb_digits)}"
            )
        ##############################################################


######################################################################

import rpl


class RPL(Task):
    def tensorize(self, sequences):
        len_max = max([len(x) for x in sequences])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [
                            self.token2id[str(c)]
                            for c in s + ["<nul>"] * (len_max - len(s))
                        ]
                        for s in sequences
                    ]
                )
            ],
            0,
        )

    def seq2str(self, seq):
        return " ".join([self.id2token[i] for i in seq])

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        nb_starting_values=3,
        max_input=9,
        prog_len=6,
        nb_runs=5,
        no_prog=False,
        logger=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.no_prog = no_prog

        train_sequences = [
            rpl.generate(
                nb_starting_values=nb_starting_values,
                nb_result_values_max=4 * nb_starting_values,
                max_input=max_input,
                prog_len=prog_len,
                nb_runs=nb_runs,
            )
            for _ in tqdm.tqdm(range(nb_train_samples), desc="train-data")
        ]

        test_sequences = [
            rpl.generate(
                nb_starting_values=nb_starting_values,
                nb_result_values_max=4 * nb_starting_values,
                max_input=max_input,
                prog_len=prog_len,
                nb_runs=nb_runs,
            )
            for _ in tqdm.tqdm(range(nb_test_samples), desc="test-data")
        ]

        symbols = list(
            set(["<nul>"] + [x for l in train_sequences + test_sequences for x in l])
        )
        val_max = max([x if type(x) is int else 0 for x in symbols])
        symbols = list(filter(lambda x: type(x) is str, symbols))
        symbols.sort()
        symbols += [str(n) for n in range(val_max + 1)]
        self.token2id = dict([(c, n) for n, c in enumerate(symbols)])
        self.id2token = dict([(n, c) for c, n in self.token2id.items()])

        self.t_nul = self.token2id["<nul>"]
        self.t_input = self.token2id["<in>"]
        self.t_output = self.token2id["<out>"]
        self.t_prog = self.token2id["<prg>"]
        self.t_end = self.token2id["<end>"]

        self.train_input = self.tensorize(train_sequences)
        self.test_input = self.tensorize(test_sequences)

        if no_prog:
            # Excise the program from every train and test example
            k = torch.arange(self.train_input.size(1), device=self.train_input.device)[
                None, :
            ]
            p = (
                ((self.train_input == self.t_prog).long() * k)
                .max(1, keepdim=True)
                .values
            )
            self.train_input = (
                self.train_input * (k <= p).long()
                + self.t_end * (k == p + 1).long()
                + self.t_nul * (k > p + 1).long()
            )
            k = torch.arange(self.test_input.size(1), device=self.test_input.device)[
                None, :
            ]
            p = (
                ((self.test_input == self.t_prog).long() * k)
                .max(1, keepdim=True)
                .values
            )
            self.test_input = (
                self.test_input * (k <= p).long()
                + self.t_end * (k == p + 1).long()
                + self.t_nul * (k > p + 1).long()
            )

        if logger is not None:
            logger(f"value_max {val_max}")
            for x in self.train_input[:25]:
                end = (x != self.t_nul).nonzero().max().item() + 1
                seq = [self.id2token[i.item()] for i in x[:end]]
                s = " ".join(seq)
                logger(f"example_seq {s}")

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            last = (batch != self.t_nul).max(0).values.nonzero().max() + 3
            batch = batch[:, :last].to(self.device)
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        # --------------------------------------------------------------------
        def compute_nb_errors_prog(input, nb_to_log=0):
            result = input.clone()
            s = (result == self.t_prog).long()
            ar_mask = (s.cumsum(dim=1) - s).clamp(min=0, max=1)
            result = (1 - ar_mask) * result + ar_mask * self.t_nul

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            sum_nb_total, sum_nb_errors = 0, 0
            for one_input, one_result in zip(input, result):
                seq = [self.id2token[i.item()] for i in one_result]
                nb_total, nb_errors, prog, stacks = rpl.compute_nb_errors(seq)
                sum_nb_total += 1
                sum_nb_errors += 0 if nb_errors == 0 else 1
                if nb_to_log > 0:
                    gt_seq = [self.id2token[i.item()] for i in one_input]
                    _, _, gt_prog, _ = rpl.compute_nb_errors(gt_seq)
                    gt_prog = " ".join([str(x) for x in gt_prog])
                    prog = " ".join([str(x) for x in prog])
                    comment = "*" if nb_errors == 0 else "-"
                    logger(f"{comment} PROG [{gt_prog}] PREDICTED [{prog}]")
                    for start_stack, target_stack, result_stack, correct in stacks:
                        comment = "*" if correct else "-"
                        start_stack = " ".join([str(x) for x in start_stack])
                        target_stack = " ".join([str(x) for x in target_stack])
                        result_stack = " ".join([str(x) for x in result_stack])
                        logger(
                            f"  {comment} [{start_stack}] -> [{target_stack}] PREDICTED [{result_stack}]"
                        )
                    nb_to_log -= 1

            return sum_nb_total, sum_nb_errors

        # --------------------------------------------------------------------
        def compute_nb_errors_output(input, nb_to_log=0):
            result = input.clone()
            k = torch.arange(result.size(1), device=result.device)[None, :]
            last_output_idx = (
                ((result == self.t_output) * k).max(dim=1, keepdim=True).values
            )
            first_prog_idx = (
                ((result == self.t_prog) * k).max(dim=1, keepdim=True).values
            )
            ar_mask = (k > last_output_idx).long() * (k < first_prog_idx).long()
            result = (1 - ar_mask) * result + ar_mask * self.t_nul

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            sum_nb_total, sum_nb_errors = 0, 0
            for one_input, one_result, i, j in zip(
                input, result, last_output_idx, first_prog_idx
            ):
                seq = [self.id2token[i.item()] for i in one_result]
                sum_nb_total += 1
                correct = (one_input - one_result).abs().max() == 0
                sum_nb_errors += 0 if correct else 1
                if nb_to_log > 0:
                    result_stack = [
                        self.id2token[i.item()] for i in one_result[i : j + 1]
                    ]
                    target_stack = [
                        self.id2token[i.item()] for i in one_input[i : j + 1]
                    ]
                    comment = "*" if correct else "-"
                    result_stack = " ".join([str(x) for x in result_stack])
                    target_stack = " ".join([str(x) for x in target_stack])
                    logger(
                        f"output_test {comment} [{target_stack}] PREDICTED [{result_stack}]"
                    )
                    nb_to_log -= 1

            return sum_nb_total, sum_nb_errors

        # --------------------------------------------------------------------

        if not self.no_prog:
            test_nb_total, test_nb_errors = compute_nb_errors_prog(
                self.test_input[:1000].to(self.device), nb_to_log=10
            )

            logger(
                f"accuracy_prog_test {n_epoch} nb_total {test_nb_total} nb_errors {test_nb_errors} accuracy {100.0*(1-test_nb_errors/test_nb_total):.02f}%"
            )

            logger(f"main_test_accuracy {n_epoch} {1-test_nb_errors/test_nb_total}")

        test_nb_total, test_nb_errors = compute_nb_errors_output(
            self.test_input[:1000].to(self.device), nb_to_log=10
        )

        logger(
            f"accuracy_output_test {n_epoch} nb_total {test_nb_total} nb_errors {test_nb_errors} accuracy {100.0*(1-test_nb_errors/test_nb_total):.02f}%"
        )

        if save_attention_image is None:
            logger("no save_attention_image (is pycairo installed?)")
        else:
            ns = torch.randint(self.test_input.size(0), (1,)).item()
            input = self.test_input[ns : ns + 1].clone()
            last = (input != self.t_nul).max(0).values.nonzero().max() + 3
            input = input[:, :last].to(self.device)

            with torch.autograd.no_grad():
                t = model.training
                model.eval()
                model.record_attention(True)
                model(BracketedSequence(input))
                model.train(t)
                ram = model.retrieve_attention()
                model.record_attention(False)

            tokens_output = [self.id2token[i.item()] for i in input[0]]
            tokens_input = ["n/a"] + tokens_output[:-1]
            for n_head in range(ram[0].size(1)):
                filename = os.path.join(
                    result_dir, f"rpl_attention_{n_epoch}_h{n_head}.pdf"
                )
                attention_matrices = [m[0, n_head] for m in ram]
                save_attention_image(
                    filename,
                    tokens_input,
                    tokens_output,
                    attention_matrices,
                    k_top=10,
                    # min_total_attention=0.9,
                    token_gap=12,
                    layer_gap=50,
                )
                logger(f"wrote {filename}")


######################################################################


import expr


class Expr(Task):
    def tensorize(self, sequences):
        len_max = max([len(x) for x in sequences])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [self.char2id[c] for c in s + "#" * (len_max - len(s))]
                        for s in sequences
                    ]
                )
            ],
            0,
        ).to(self.device)

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        nb_variables,
        sequence_length,
        operand_max,
        result_max,
        batch_size,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device

        train_sequences = expr.generate_sequences(
            nb_train_samples,
            nb_variables=nb_variables,
            length=sequence_length,
            operand_max=operand_max,
            result_max=result_max,
        )

        test_sequences = expr.generate_sequences(
            nb_test_samples,
            nb_variables=nb_variables,
            length=sequence_length,
            operand_max=operand_max,
            result_max=result_max,
        )

        symbols = list(set("#" + "".join(train_sequences + test_sequences)))
        symbols.sort()

        self.char2id = dict([(c, n) for n, c in enumerate(symbols)])
        self.id2char = dict([(n, c) for c, n in self.char2id.items()])

        self.filler, self.space = self.char2id["#"], self.char2id[" "]

        self.train_input = self.tensorize(train_sequences)
        self.test_input = self.tensorize(test_sequences)

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            last = (batch != self.filler).max(0).values.nonzero().max() + 3
            batch = batch[:, :last]
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def seq2str(self, s):
        return "".join([self.id2char[k.item()] for k in s])

    def produce_results(
        self,
        n_epoch,
        model,
        result_dir,
        logger,
        deterministic_synthesis,
        input_file=None,
    ):
        def compute_nb_correct(input):
            result = input.clone()
            s = (result == self.space).long()
            ar_mask = (s.cumsum(dim=1) - s).clamp(min=0, max=1)
            result = (1 - ar_mask) * result + ar_mask * self.filler
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            nb_total = input.size(0)
            nb_correct = (input == result).long().min(1).values.sum()

            #######################################################################
            # Comput predicted vs. true variable values

            nb_delta = torch.zeros(5, dtype=torch.int64)
            nb_missed = 0

            values_input = expr.extract_results([self.seq2str(s) for s in input])
            values_result = expr.extract_results([self.seq2str(s) for s in result])

            filename = os.path.join(result_dir, f"expr_result_{n_epoch:04d}.txt")

            with open(filename, "w") as f:
                for i, r in zip(values_input, values_result):
                    for n, vi in i.items():
                        vr = r.get(n)
                        f.write(f"{vi} {-1 if vr is None else vr}\n")

                        if vr is None or vr < 0:
                            nb_missed += 1
                        else:
                            d = abs(vr - vi)
                            if d >= nb_delta.size(0):
                                nb_missed += 1
                            else:
                                nb_delta[d] += 1

            ######################################################################

            return nb_total, nb_correct, nb_delta, nb_missed

        (
            test_nb_total,
            test_nb_correct,
            test_nb_delta,
            test_nb_missed,
        ) = compute_nb_correct(self.test_input[:10000])

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )

        logger(f"main_test_accuracy {n_epoch} {test_nb_correct/test_nb_total}")

        nb_total = test_nb_delta.sum() + test_nb_missed
        for d in range(test_nb_delta.size(0)):
            logger(
                f"error_value {n_epoch} delta {d} {test_nb_delta[d]} {test_nb_delta[d]*100/nb_total:.02f}%"
            )
        logger(
            f"error_value {n_epoch} missed {test_nb_missed} {test_nb_missed*100/nb_total:.02f}%"
        )

        ##############################################################
        # Log a few generated sequences
        if input_file is None:
            input = self.test_input[:10]
        else:
            with open(input_file, "r") as f:
                sequences = [e.strip() for e in f.readlines()]
                sequences = [s + " " + "#" * 50 for s in sequences]
                input = self.tensorize(sequences)

        result = input.clone()
        s = (result == self.space).long()
        ar_mask = (s.cumsum(dim=1) - s).clamp(min=0, max=1)
        result = (1 - ar_mask) * result + ar_mask * self.filler

        for n in range(result.size(0)):
            logger(f"test_before {self.seq2str(result[n])}")

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        correct = (1 - ar_mask) * self.space + ar_mask * input
        for n in range(result.size(0)):
            comment = "GOOD" if (result[n] - input[n]).abs().max() == 0 else ""
            logger(f"test_after  {self.seq2str(result[n])} {comment}")
            logger(f"truth       {self.seq2str(correct[n])}")
        ##############################################################


######################################################################

import grid


class Grid(Task):
    # Make a tensor from a list of strings
    def str2tensor(self, descr):
        token_descr = [s.strip().split(" ") for s in descr]
        l = max([len(s) for s in token_descr])
        token_descr = [s + ["#"] * (l - len(s)) for s in token_descr]
        id_descr = [[self.token2id[u] for u in s] for s in token_descr]
        return torch.tensor(id_descr, device=self.device)

    # Make a list of strings from a tensor
    def tensor2str(self, x):
        return [" ".join([self.id2token[t.item()] for t in r]) for r in x]

    # trim all the tensors in the tuple z to remove as much token from
    # left and right in the first tensor. If z is a tuple, all its
    # elements are trimed according to the triming for the first
    def trim(self, z, token="#"):
        n = self.token2id[token]
        if type(z) == tuple:
            x = z[0]
            i = (1 - (F.pad(x, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return tuple([t[:, a:b] for t in z])
        else:
            i = (1 - (F.pad(z, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return z[:, a:b]

    ######################

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        size,
        fraction_play=0.0,
        logger=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.grid_factory = grid.GridFactory(size=size)
        self.fraction_play = fraction_play

        if logger is not None:
            logger(
                f"generating {nb_train_samples+nb_test_samples} samples (can take some time)"
            )

        self.train_descr = self.grid_factory.generate_samples(
            nb=nb_train_samples,
            fraction_play=fraction_play,
            progress_bar=lambda r: tqdm.tqdm(r),
        )

        self.test_descr = self.grid_factory.generate_samples(
            nb=nb_test_samples, fraction_play=0.0, progress_bar=lambda r: tqdm.tqdm(r)
        )

        if fraction_play > 0:
            self.play_descr = self.grid_factory.generate_samples(
                nb=25, fraction_play=1.0, progress_bar=lambda r: tqdm.tqdm(r)
            )
        else:
            self.play_descr = []

        # Build the tokenizer
        tokens = set()
        for d in [self.train_descr, self.test_descr, self.play_descr]:
            for s in d:
                for t in s.strip().split(" "):
                    tokens.add(t)
        # make this set a sorted list to get the same tensors given
        # the same descr
        tokens = list(tokens)
        tokens.sort()
        tokens = ["#"] + tokens
        self.token2id = dict([(t, n) for n, t in enumerate(tokens)])
        self.id2token = dict([(n, t) for n, t in enumerate(tokens)])
        self.t_nul = self.token2id["#"]
        self.t_true = self.token2id["true"]
        self.t_false = self.token2id["false"]
        self.t_pipe = self.token2id["|"]

        # Tokenize the train and test sets
        self.train_input = self.str2tensor(self.train_descr)
        self.test_input = self.str2tensor(self.test_descr)
        self.play_input = (
            None if len(self.play_descr) == 0 else self.str2tensor(self.play_descr)
        )

    def batches(self, split="train"):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=f"epoch-{split}"
        ):
            yield self.trim(batch)

    def vocabulary_size(self):
        return len(self.token2id)

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        correct = self.test_input[:1000]
        result = correct.clone()
        ar_mask = torch.logical_or(result == self.t_true, result == self.t_false).long()
        result *= 1 - ar_mask  # paraaaaanoiaaaaaaa

        logger(f"----------------------------------------------------------")

        for e in self.tensor2str(result[:10]):
            logger(f"test_before {e}")

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        logger(f"----------------------------------------------------------")

        for e in self.tensor2str(result[:10]):
            logger(f"test_after  {e}")

        logger(f"----------------------------------------------------------")

        nb_total = ar_mask.sum().item()
        nb_correct = ((correct == result).long() * ar_mask).sum().item()

        logger(f"test_performance {n_epoch} {nb_total=} {nb_correct=}")
        logger(f"main_test_accuracy {n_epoch} {nb_correct / nb_total}")

        if self.play_input is not None:
            result = self.play_input.clone()
            ar_mask = (result == self.t_pipe).long().cumsum(dim=1).clamp(max=1)
            result *= 1 - ar_mask  # paraaaaanoiaaaaaaa

            logger(f"----------------------------------------------------------")

            for e in self.tensor2str(result[:10]):
                logger(f"play_before {e}")

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            logger(f"----------------------------------------------------------")

            for e in self.tensor2str(result[:10]):
                logger(f"play_after  {e}")

            logger(f"----------------------------------------------------------")


######################################################################

import qmlp


class QMLP(Task):
    ######################

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        result_dir,
        logger=None,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.nb_samples_per_mlp = 256

        if logger is not None:
            logger(
                f"generating {nb_train_samples+nb_test_samples} samples (can take some time)"
            )

        seq, q_test_set, test_error = qmlp.generate_sequence_and_test_set(
            nb_mlps=nb_train_samples + nb_test_samples,
            nb_samples=self.nb_samples_per_mlp,
            device=self.device,
            batch_size=64,
            nb_epochs=250,
            nb_mlps_per_batch=1024,
        )

        self.train_input = seq[:nb_train_samples]
        self.train_q_test_set = q_test_set[:nb_train_samples]
        self.train_ref_test_errors = test_error[:nb_train_samples]
        self.test_input = seq[nb_train_samples:]
        self.test_q_test_set = q_test_set[nb_train_samples:]
        self.test_ref_test_errors = test_error[nb_train_samples:]

        filename = os.path.join(result_dir, f"train_errors_ref.dat")
        with open(filename, "w") as f:
            for e in self.train_ref_test_errors:
                f.write(f"{e}\n")

        filename = os.path.join(result_dir, f"test_errors_ref.dat")
        with open(filename, "w") as f:
            for e in self.test_ref_test_errors:
                f.write(f"{e}\n")

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train"):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=f"epoch-{split}"
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        correct = self.test_input[:1000]
        result = correct.clone()
        ar_mask = (
            torch.arange(result.size(1), device=result.device)
            > self.nb_samples_per_mlp * 3 + 1
        ).long()[None, :]
        ar_mask = ar_mask.expand_as(result)
        result *= 1 - ar_mask  # paraaaaanoiaaaaaaa

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        q_train_set = result[:, : self.nb_samples_per_mlp * 3]
        q_params = result[:, self.nb_samples_per_mlp * 3 + 1 :]
        error_test = qmlp.evaluate_q_params(q_params, self.test_q_test_set)

        filename = os.path.join(result_dir, f"test_errors_{n_epoch:04d}.dat")
        with open(filename, "w") as f:
            for e in error_test:
                f.write(f"{e}\n")


######################################################################
