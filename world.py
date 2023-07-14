#!/usr/bin/env python

import math, sys, tqdm

import torch, torchvision

from torch import nn
from torch.nn import functional as F
import cairo

######################################################################


class Box:
    nb_rgb_levels = 10

    def __init__(self, x, y, w, h, r, g, b):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.r = r
        self.g = g
        self.b = b

    def collision(self, scene):
        for c in scene:
            if (
                self is not c
                and max(self.x, c.x) <= min(self.x + self.w, c.x + c.w)
                and max(self.y, c.y) <= min(self.y + self.h, c.y + c.h)
            ):
                return True
        return False


######################################################################


class Normalizer(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("log_var", 2 * torch.log(std))

    def forward(self, x):
        return (x - self.mu) / torch.exp(self.log_var / 2.0)


class SignSTE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # torch.sign() takes three values
        s = (x >= 0).float() * 2 - 1

        if self.training:
            u = torch.tanh(x)
            return s + u - u.detach()
        else:
            return s


def train_encoder(
    train_input,
    test_input,
    depth=3,
    dim_hidden=48,
    nb_bits_per_token=8,
    lr_start=1e-3,
    lr_end=1e-4,
    nb_epochs=10,
    batch_size=25,
    device=torch.device("cpu"),
):
    mu, std = train_input.float().mean(), train_input.float().std()

    def encoder_core(depth, dim):
        l = [
            [
                nn.Conv2d(
                    dim * 2**k, dim * 2**k, kernel_size=5, stride=1, padding=2
                ),
                nn.ReLU(),
                nn.Conv2d(dim * 2**k, dim * 2 ** (k + 1), kernel_size=2, stride=2),
                nn.ReLU(),
            ]
            for k in range(depth)
        ]

        return nn.Sequential(*[x for m in l for x in m])

    def decoder_core(depth, dim):
        l = [
            [
                nn.ConvTranspose2d(
                    dim * 2 ** (k + 1), dim * 2**k, kernel_size=2, stride=2
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    dim * 2**k, dim * 2**k, kernel_size=5, stride=1, padding=2
                ),
                nn.ReLU(),
            ]
            for k in range(depth - 1, -1, -1)
        ]

        return nn.Sequential(*[x for m in l for x in m])

    encoder = nn.Sequential(
        Normalizer(mu, std),
        nn.Conv2d(3, dim_hidden, kernel_size=1, stride=1),
        nn.ReLU(),
        # 64x64
        encoder_core(depth=depth, dim=dim_hidden),
        # 8x8
        nn.Conv2d(dim_hidden * 2**depth, nb_bits_per_token, kernel_size=1, stride=1),
    )

    quantizer = SignSTE()

    decoder = nn.Sequential(
        nn.Conv2d(nb_bits_per_token, dim_hidden * 2**depth, kernel_size=1, stride=1),
        # 8x8
        decoder_core(depth=depth, dim=dim_hidden),
        # 64x64
        nn.ConvTranspose2d(dim_hidden, 3 * Box.nb_rgb_levels, kernel_size=1, stride=1),
    )

    model = nn.Sequential(encoder, decoder)

    nb_parameters = sum(p.numel() for p in model.parameters())

    print(f"nb_parameters {nb_parameters}")

    model.to(device)

    for k in range(nb_epochs):
        lr = math.exp(
            math.log(lr_start) + math.log(lr_end / lr_start) / (nb_epochs - 1) * k
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        acc_train_loss = 0.0

        for input in tqdm.tqdm(train_input.split(batch_size), desc="vqae-train"):
            z = encoder(input)
            zq = z if k < 2 else quantizer(z)
            output = decoder(zq)

            output = output.reshape(
                output.size(0), -1, 3, output.size(2), output.size(3)
            )

            train_loss = F.cross_entropy(output, input)

            acc_train_loss += train_loss.item() * input.size(0)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        acc_test_loss = 0.0

        for input in tqdm.tqdm(test_input.split(batch_size), desc="vqae-test"):
            z = encoder(input)
            zq = z if k < 1 else quantizer(z)
            output = decoder(zq)

            output = output.reshape(
                output.size(0), -1, 3, output.size(2), output.size(3)
            )

            test_loss = F.cross_entropy(output, input)

            acc_test_loss += test_loss.item() * input.size(0)

        train_loss = acc_train_loss / train_input.size(0)
        test_loss = acc_test_loss / test_input.size(0)

        print(f"train_ae {k} lr {lr} train_loss {train_loss} test_loss {test_loss}")
        sys.stdout.flush()

    return encoder, quantizer, decoder


######################################################################


def scene2tensor(xh, yh, scene, size):
    width, height = size, size
    pixel_map = torch.ByteTensor(width, height, 4).fill_(255)
    data = pixel_map.numpy()
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_ARGB32, width, height
    )

    ctx = cairo.Context(surface)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

    for b in scene:
        ctx.move_to(b.x * size, b.y * size)
        ctx.rel_line_to(b.w * size, 0)
        ctx.rel_line_to(0, b.h * size)
        ctx.rel_line_to(-b.w * size, 0)
        ctx.close_path()
        ctx.set_source_rgba(
            b.r / (Box.nb_rgb_levels - 1),
            b.g / (Box.nb_rgb_levels - 1),
            b.b / (Box.nb_rgb_levels - 1),
            1.0,
        )
        ctx.fill()

    hs = size * 0.1
    ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    ctx.move_to(xh * size - hs / 2, yh * size - hs / 2)
    ctx.rel_line_to(hs, 0)
    ctx.rel_line_to(0, hs)
    ctx.rel_line_to(-hs, 0)
    ctx.close_path()
    ctx.fill()

    return (
        pixel_map[None, :, :, :3]
        .flip(-1)
        .permute(0, 3, 1, 2)
        .long()
        .mul(Box.nb_rgb_levels)
        .floor_divide(256)
    )


def random_scene():
    scene = []
    colors = [
        ((Box.nb_rgb_levels - 1), 0, 0),
        (0, (Box.nb_rgb_levels - 1), 0),
        (0, 0, (Box.nb_rgb_levels - 1)),
        ((Box.nb_rgb_levels - 1), (Box.nb_rgb_levels - 1), 0),
        (
            (Box.nb_rgb_levels * 2) // 3,
            (Box.nb_rgb_levels * 2) // 3,
            (Box.nb_rgb_levels * 2) // 3,
        ),
    ]

    for k in range(10):
        wh = torch.rand(2) * 0.2 + 0.2
        xy = torch.rand(2) * (1 - wh)
        c = colors[torch.randint(len(colors), (1,))]
        b = Box(
            xy[0].item(), xy[1].item(), wh[0].item(), wh[1].item(), c[0], c[1], c[2]
        )
        if not b.collision(scene):
            scene.append(b)

    return scene


def generate_episode(steps, size=64):
    delta = 0.1
    effects = [
        (False, 0, 0),
        (False, delta, 0),
        (False, 0, delta),
        (False, -delta, 0),
        (False, 0, -delta),
        (True, delta, 0),
        (True, 0, delta),
        (True, -delta, 0),
        (True, 0, -delta),
    ]

    while True:
        frames = []

        scene = random_scene()
        xh, yh = tuple(x.item() for x in torch.rand(2))

        actions = torch.randint(len(effects), (len(steps),))
        change = False

        for s, a in zip(steps, actions):
            if s:
                frames.append(scene2tensor(xh, yh, scene, size=size))

            g, dx, dy = effects[a]
            if g:
                for b in scene:
                    if b.x <= xh and b.x + b.w >= xh and b.y <= yh and b.y + b.h >= yh:
                        x, y = b.x, b.y
                        b.x += dx
                        b.y += dy
                        if (
                            b.x < 0
                            or b.y < 0
                            or b.x + b.w > 1
                            or b.y + b.h > 1
                            or b.collision(scene)
                        ):
                            b.x, b.y = x, y
                        else:
                            xh += dx
                            yh += dy
                            change = True
            else:
                x, y = xh, yh
                xh += dx
                yh += dy
                if xh < 0 or xh > 1 or yh < 0 or yh > 1:
                    xh, yh = x, y

        if change:
            break

    return frames, actions


######################################################################


def generate_episodes(nb, steps):
    all_frames = []
    for n in tqdm.tqdm(range(nb), dynamic_ncols=True, desc="world-data"):
        frames, actions = generate_episode(steps)
        all_frames += frames
    return torch.cat(all_frames, 0).contiguous()


def create_data_and_processors(nb_train_samples, nb_test_samples, nb_epochs=10):
    steps = [True] + [False] * 30 + [True]
    train_input = generate_episodes(nb_train_samples, steps)
    test_input = generate_episodes(nb_test_samples, steps)

    encoder, quantizer, decoder = train_encoder(
        train_input, test_input, nb_epochs=nb_epochs
    )
    encoder.train(False)
    quantizer.train(False)
    decoder.train(False)

    z = encoder(train_input[:1])
    pow2 = (2 ** torch.arange(z.size(1), device=z.device))[None, None, :]
    z_h, z_w = z.size(2), z.size(3)

    def frame2seq(input, batch_size=25):
        seq = []

        for x in input.split(batch_size):
            z = encoder(x)
            ze_bool = (quantizer(z) >= 0).long()
            output = (
                ze_bool.permute(0, 2, 3, 1).reshape(
                    ze_bool.size(0), -1, ze_bool.size(1)
                )
                * pow2
            ).sum(-1)

            seq.append(output)

        return torch.cat(seq, dim=0)

    def seq2frame(input, batch_size=25, T=1e-2):
        frames = []

        for seq in input.split(batch_size):
            zd_bool = (seq[:, :, None] // pow2) % 2
            zd_bool = zd_bool.reshape(zd_bool.size(0), z_h, z_w, -1).permute(0, 3, 1, 2)
            logits = decoder(zd_bool * 2.0 - 1.0)
            logits = logits.reshape(
                logits.size(0), -1, 3, logits.size(2), logits.size(3)
            ).permute(0, 2, 3, 4, 1)
            output = torch.distributions.categorical.Categorical(
                logits=logits / T
            ).sample()

            frames.append(output)

        return torch.cat(frames, dim=0)

    return train_input, test_input, frame2seq, seq2frame


######################################################################

if __name__ == "__main__":
    train_input, test_input, frame2seq, seq2frame = create_data_and_processors(
        # 10000, 1000,
        100,
        100,
        nb_epochs=2,
    )

    input = test_input[:64]

    seq = frame2seq(input)

    print(f"{seq.size()=} {seq.dtype=} {seq.min()=} {seq.max()=}")

    output = seq2frame(seq)

    torchvision.utils.save_image(
        input.float() / (Box.nb_rgb_levels - 1), "orig.png", nrow=8
    )

    torchvision.utils.save_image(
        output.float() / (Box.nb_rgb_levels - 1), "qtiz.png", nrow=8
    )
