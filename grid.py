#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math
import torch, torchvision
import torch.nn.functional as F

######################################################################


class GridFactory:
    def __init__(
        self,
        size=6,
        max_nb_items=4,
        max_nb_transformations=3,
        nb_questions=4,
        nb_shapes=6,
        nb_colors=6,
        nb_play_steps=3,
    ):
        assert size % 2 == 0
        self.size = size
        self.max_nb_items = max_nb_items
        self.max_nb_transformations = max_nb_transformations
        self.nb_questions = nb_questions
        self.nb_play_steps = nb_play_steps
        self.name_shapes = ["A", "B", "C", "D", "E", "F"]
        self.name_colors = ["red", "yellow", "blue", "green", "white", "purple"]
        self.vname_shapes = ["vA", "vB", "vC", "vD", "vE", "vF"]
        self.vname_colors = ["vred", "vyellow", "vblue", "vgreen", "vwhite", "vpurple"]

    def generate_scene(self):
        nb_items = torch.randint(self.max_nb_items - 1, (1,)).item() + 2
        col = torch.full((self.size * self.size,), -1)
        shp = torch.full((self.size * self.size,), -1)
        a = torch.randperm(len(self.name_colors) * len(self.name_shapes))[:nb_items]
        col[:nb_items] = a % len(self.name_colors)
        shp[:nb_items] = a // len(self.name_colors)
        i = torch.randperm(self.size * self.size)
        col = col[i]
        shp = shp[i]
        return col.reshape(self.size, self.size), shp.reshape(self.size, self.size)

    def random_object_move(self, scene):
        col, shp = scene
        while True:
            a = (col.flatten() >= 0).nonzero()
            a = a[torch.randint(a.size(0), (1,)).item()]
            i, j = a // self.size, a % self.size
            assert col[i, j] >= 0
            dst = [(i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            dst = list(
                filter(
                    lambda x: x[0] >= 0
                    and x[1] >= 0
                    and x[0] < self.size
                    and x[1] < self.size
                    and col[x[0], x[1]] < 0,
                    dst,
                )
            )
            if len(dst) > 0:
                ni, nj = dst[torch.randint(len(dst), (1,)).item()]
                col[ni, nj] = col[i, j]
                shp[ni, nj] = shp[i, j]
                col[i, j] = -1
                shp[i, j] = -1
                break

        return col, shp

    def transformation(self, t, scene):
        col, shp = scene
        if t == 0:
            col, shp = col.flip(0), shp.flip(0)
            description = "<chg> vertical flip"
        elif t == 1:
            col, shp = col.flip(1), shp.flip(1)
            description = "<chg> horizontal flip"
        elif t == 2:
            col, shp = col.flip(0).t(), shp.flip(0).t()
            description = "<chg> rotate 90 degrees"
        elif t == 3:
            col, shp = col.flip(0).flip(1), shp.flip(0).flip(1)
            description = "<chg> rotate 180 degrees"
        elif t == 4:
            col, shp = col.flip(1).t(), shp.flip(1).t()
            description = "<chg> rotate 270 degrees"

        return (col.contiguous(), shp.contiguous()), description

    def random_transformations(self, scene):
        descriptions = []
        nb_transformations = torch.randint(self.max_nb_transformations + 1, (1,)).item()
        transformations = torch.randint(5, (nb_transformations,))

        for t in transformations:
            scene, description = self.transformation(t, scene)
            descriptions += [description]

        return scene, descriptions

    def visual_scene2str(self, scene):
        col, shp = scene
        r = []
        for i in range(self.size):
            s = []
            for j in range(self.size):
                if col[i, j] >= 0:
                    s += [self.vname_colors[col[i, j]], self.vname_shapes[shp[i, j]]]
                else:
                    s += ["v_", "v+"]
            r += s  # .append(" ".join(s))
        return " ".join(r)

    def print_scene(self, scene):
        col, shp = scene

        # for i in range(self.size):
        # for j in range(self.size):
        # if col[i,j] >= 0:
        # print(f"at ({i},{j}) {self.name_colors[col[i,j]]} {self.name_shapes[shp[i,j]]}")

        for i in range(self.size):
            for j in range(self.size):
                if col[i, j] >= 0:
                    print(
                        f"{self.name_colors[col[i,j]][0]}{self.name_shapes[shp[i,j]]}",
                        end="",
                    )
                elif j == 0:
                    print(" +", end="")
                else:
                    print("-+", end="")
                if j < self.size - 1:
                    print("--", end="")
                else:
                    print("")
            if i < self.size - 1:
                for j in range(self.size - 1):
                    print(" |  ", end="")
                print(" |")

    def grid_positions(self, scene):
        col, shp = scene

        properties = []

        for i in range(self.size):
            for j in range(self.size):
                if col[i, j] >= 0:
                    n = f"{self.name_colors[col[i,j]]} {self.name_shapes[shp[i,j]]}"
                    properties += [f"a {n} at {i} {j}"]

        return properties

    def all_properties(self, scene):
        col, shp = scene

        properties = []

        for i1 in range(self.size):
            for j1 in range(self.size):
                if col[i1, j1] >= 0:
                    n1 = (
                        f"{self.name_colors[col[i1,j1]]} {self.name_shapes[shp[i1,j1]]}"
                    )
                    properties += [f"there is a {n1}"]
                    if i1 < self.size // 2:
                        properties += [f"a {n1} is in the top half"]
                    if i1 >= self.size // 2:
                        properties += [f"a {n1} is in the bottom half"]
                    if j1 < self.size // 2:
                        properties += [f"a {n1} is in the left half"]
                    if j1 >= self.size // 2:
                        properties += [f"a {n1} is in the right half"]
                    for i2 in range(self.size):
                        for j2 in range(self.size):
                            if col[i2, j2] >= 0:
                                n2 = f"{self.name_colors[col[i2,j2]]} {self.name_shapes[shp[i2,j2]]}"
                                if i1 > i2:
                                    properties += [f"a {n1} is below a {n2}"]
                                if i1 < i2:
                                    properties += [f"a {n1} is above a {n2}"]
                                if j1 > j2:
                                    properties += [f"a {n1} is right of a {n2}"]
                                if j1 < j2:
                                    properties += [f"a {n1} is left of a {n2}"]
                                if abs(i1 - i2) + abs(j1 - j2) == 1:
                                    properties += [f"a {n1} is next to a {n2}"]

        return properties

    def generate_scene_and_play(self):
        scene = self.generate_scene()
        steps = [self.visual_scene2str(scene)]
        for t in range(self.nb_play_steps - 1):
            if torch.randint(4, (1,)).item() == 0:
                scene, _ = self.transformation(torch.randint(5, (1,)), scene)
            else:
                scene = self.random_object_move(scene)
            steps.append(self.visual_scene2str(scene))
        return " | ".join(steps)

    def generate_scene_and_questions(self):
        while True:
            # We generate scenes until we get one with enough
            # properties

            while True:
                start_scene = self.generate_scene()
                scene, transformations = self.random_transformations(start_scene)
                true = self.all_properties(scene)
                if len(true) >= self.nb_questions:
                    break

            # We generate a bunch of false properties by shuffling the
            # scene and sometimes adding properties from totally
            # different scenes. We try ten times to get enough false
            # properties and go back to generating the scene if we do
            # not succeed

            for a in range(10):
                col, shp = scene
                col, shp = col.view(-1), shp.view(-1)
                p = torch.randperm(col.size(0))
                col, shp = col[p], shp[p]
                other_scene = (
                    col.view(self.size, self.size),
                    shp.view(self.size, self.size),
                )

                false = self.all_properties(other_scene)

                # We sometime add properties from a totally different
                # scene to have negative "there is a xxx xxx"
                # properties

                if torch.rand(1).item() < 0.2:
                    other_scene = self.generate_scene()
                    false += self.all_properties(other_scene)

                false = list(set(false) - set(true))
                if len(false) >= self.nb_questions:
                    break

            if a < 10:
                break

        true = [true[k] for k in torch.randperm(len(true))[: self.nb_questions]]
        false = [false[k] for k in torch.randperm(len(false))[: self.nb_questions]]
        true = ["<prop> " + q + " <ans> true" for q in true]
        false = ["<prop> " + q + " <ans> false" for q in false]

        union = true + false
        questions = [union[k] for k in torch.randperm(len(union))[: self.nb_questions]]

        result = " ".join(
            ["<obj> " + x for x in self.grid_positions(start_scene)]
            + transformations
            + questions
        )

        return start_scene, scene, result

    def generate_samples(self, nb, fraction_play=0.0, progress_bar=None):
        result = []

        play = torch.rand(nb) < fraction_play
        if progress_bar is not None:
            play = progress_bar(play)

        for p in play:
            if p:
                result.append(self.generate_scene_and_play())
            else:
                result.append(self.generate_scene_and_questions()[2])

        return result


######################################################################

if __name__ == "__main__":
    import time

    grid_factory = GridFactory()

    # start_time = time.perf_counter()
    # samples = grid_factory.generate_samples(10000)
    # end_time = time.perf_counter()
    # print(f"{len(samples) / (end_time - start_time):.02f} samples per second")

    start_scene, scene, questions = grid_factory.generate_scene_and_questions()
    print()
    print("-- Original scene -----------------------------")
    print()
    grid_factory.print_scene(start_scene)
    print()
    print("-- Transformed scene --------------------------")
    print()
    grid_factory.print_scene(scene)
    print()
    print("-- Sequence -----------------------------------")
    print()
    print(questions)

    # print(grid_factory.visual_scene2str(scene))

    # grid_factory.print_scene(scene)
    # for t in range(5):
    # scene = grid_factory.random_object_move(scene)
    # print()
    # grid_factory.print_scene(scene)

    print(grid_factory.generate_scene_and_play())

######################################################################
