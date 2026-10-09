"""External behavior of an individual labeled student-teacher."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn import metrics

from experiments.runner import (
    TEACHER_MOMENTUM,
    Collection,
    Observation,
    initial_student_teacher,
    run_experiment,
)


def observation(value, labels, augment=None):
    image = torch.full((1, 2, 2), float(value))
    augmented = None if augment is None else torch.full((1, 2, 2), float(augment))
    return Observation(image=image, labels=dict(labels), augmented=augmented)


def labeled_collection(name="fake"):
    fit = (
        observation(0.1, {"opacity": 1, "effusion": 0}, augment=0.6),
        observation(0.2, {"opacity": 0, "effusion": 1}, augment=0.7),
        observation(0.3, {"opacity": 1, "effusion": 1}, augment=0.8),
        observation(0.4, {"opacity": 0, "effusion": 0}, augment=0.9),
    )
    validation = (
        observation(0.15, {"opacity": 1, "effusion": 0}),
        observation(0.35, {"opacity": 0, "effusion": 1}),
        observation(0.55, {"opacity": 1, "effusion": 1}),
        observation(0.75, {"opacity": 0, "effusion": 0}),
    )
    return Collection(name, ("opacity", "effusion"), fit, validation)


def run(root, collection, name, **kwargs):
    settings = {"epochs": 1, "learning_rate": 0.1, "batch_size": 8, "seed": 0}
    settings.update(kwargs)
    return run_experiment(name, "individual", [collection], root, **settings)


def load_checkpoint(root, name):
    path = Path(root) / name / "checkpoint.pth"
    return torch.load(path, map_location="cpu", weights_only=False)


def load_resume(root, name):
    path = Path(root) / name / "resume.pth"
    return torch.load(path, map_location="cpu", weights_only=False)


def group_delta(before, after, prefix):
    deltas = []
    for key, value in before.items():
        if key.startswith(prefix):
            deltas.append((value - after[key]).abs().max().item())
    return max(deltas)


def oracle_scores(probabilities, observations, label_names, uncertain_as_negative=False):
    roc_auc = {}
    average_precision = {}
    for column, label_name in enumerate(label_names):
        y_true = []
        y_score = []
        for row, item in enumerate(observations):
            if label_name not in item.labels:
                y_true.append(0.0)
                y_score.append(float(probabilities[row, column]))
                continue
            value = item.labels[label_name]
            if value is None and not uncertain_as_negative:
                continue
            y_true.append(0.0 if value is None else float(value))
            y_score.append(float(probabilities[row, column]))
        roc_auc[label_name] = float(metrics.roc_auc_score(y_true, y_score))
        average_precision[label_name] = float(metrics.average_precision_score(y_true, y_score))
    headline = float(sum(roc_auc.values()) / len(roc_auc))
    macro = float(sum(average_precision.values()) / len(average_precision))
    return headline, roc_auc, macro, average_precision


class IndividualLabeledStudentTeacherTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def test_one_experiment_writes_one_metrics_row_for_one_epoch(self):
        collection = labeled_collection()
        metrics = run(self.root, collection, "alpha")
        self.assertEqual(len(metrics), 1)
        row = metrics[0]
        self.assertEqual(row["collection"], "fake")
        self.assertEqual(row["epoch"], 0)
        for field in (
            "training_loss",
            "validation_loss",
            "headline_score",
            "roc_auc",
            "macro_average_precision",
            "average_precision",
        ):
            self.assertIn(field, row)
        self.assertEqual(set(row["roc_auc"]), {"opacity", "effusion"})
        self.assertEqual(set(row["average_precision"]), {"opacity", "effusion"})
        self.assertAlmostEqual(row["headline_score"], sum(row["roc_auc"].values()) / 2)
        self.assertAlmostEqual(row["macro_average_precision"], sum(row["average_precision"].values()) / 2)
        written = json.loads((self.root / "alpha" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(written), 1)
        self.assertAlmostEqual(written[0]["headline_score"], row["headline_score"])

    def test_label_loss_updates_encoder_consistency_head_and_task_head(self):
        collection = labeled_collection()
        collection.fit = tuple(observation(item.image.flatten()[0].item(), item.labels) for item in collection.fit)
        student, _teacher = initial_student_teacher(collection, seed=0)
        before = {key: value.clone() for key, value in student.state_dict().items()}
        run(self.root, collection, "label-loss", learning_rate=0.2, batch_size=8)
        after = load_checkpoint(self.root, "label-loss")["student"]
        for prefix in ("encoder.", "consistency_head.", "task_head."):
            self.assertGreater(group_delta(before, after, prefix), 0.0)

    def test_consistency_loss_is_mean_squared_error_and_uncertain_adds_none(self):
        fit = (
            observation(0.2, {"opacity": None, "effusion": None}, augment=0.8),
            observation(0.4, {"opacity": None, "effusion": None}, augment=1.1),
        )
        validation = labeled_collection().validation
        collection = Collection("fake", ("opacity", "effusion"), fit, validation)
        student, teacher = initial_student_teacher(collection, seed=0)
        before = {key: value.clone() for key, value in student.state_dict().items()}
        augmented = torch.stack([item.augmented for item in fit])
        original = torch.stack([item.image for item in fit])
        student.train()
        teacher.eval()
        with torch.no_grad():
            expected = float(F.mse_loss(student(augmented)[0], teacher(original)))
        metrics = run(self.root, collection, "consistency", learning_rate=0.2, batch_size=8)
        self.assertAlmostEqual(metrics[0]["training_loss"], expected, places=6)
        after = load_checkpoint(self.root, "consistency")["student"]
        self.assertEqual(group_delta(before, after, "task_head."), 0.0)
        self.assertGreater(group_delta(before, after, "encoder."), 0.0)
        self.assertGreater(group_delta(before, after, "consistency_head."), 0.0)

    def test_unmentioned_observation_counts_as_negative(self):
        omitted = (
            observation(0.2, {"opacity": 1}, augment=0.5),
            observation(0.5, {"opacity": 0}, augment=0.9),
        )
        explicit = (
            observation(0.2, {"opacity": 1, "effusion": 0}, augment=0.5),
            observation(0.5, {"opacity": 0, "effusion": 0}, augment=0.9),
        )
        validation = labeled_collection().validation
        omitted_collection = Collection("fake", ("opacity", "effusion"), omitted, validation)
        explicit_collection = Collection("fake", ("opacity", "effusion"), explicit, validation)
        omitted_metrics = run(self.root, omitted_collection, "omitted")
        explicit_metrics = run(self.root, explicit_collection, "explicit")
        self.assertEqual(omitted_metrics, explicit_metrics)
        omitted_checkpoint = load_checkpoint(self.root, "omitted")
        explicit_checkpoint = load_checkpoint(self.root, "explicit")
        for key in omitted_checkpoint["student"]:
            self.assertTrue(torch.equal(omitted_checkpoint["student"][key], explicit_checkpoint["student"][key]))

    def test_uncertain_observation_is_left_out_of_roc_auc(self):
        validation = (
            observation(0.1, {"opacity": 1, "effusion": 0}),
            observation(0.3, {"opacity": 0, "effusion": 1}),
            observation(0.6, {"opacity": 1, "effusion": 1}),
            observation(0.8, {"opacity": 0, "effusion": 0}),
            observation(2.5, {"opacity": None, "effusion": 1}),
        )
        collection = Collection("fake", ("opacity", "effusion"), labeled_collection().fit, validation)
        metrics = run(self.root, collection, "uncertain", learning_rate=0.05)
        student, _teacher = initial_student_teacher(collection, seed=0)
        student.load_state_dict(load_checkpoint(self.root, "uncertain")["student"])
        probabilities = student.probabilities(torch.stack([item.image for item in validation]))
        headline, roc_auc, macro, average_precision = oracle_scores(probabilities, validation, collection.label_names)
        treated = oracle_scores(probabilities, validation, collection.label_names, uncertain_as_negative=True)
        self.assertNotEqual(roc_auc["opacity"], treated[1]["opacity"])
        row = metrics[0]
        self.assertAlmostEqual(row["headline_score"], headline)
        self.assertAlmostEqual(row["roc_auc"]["opacity"], roc_auc["opacity"])
        self.assertAlmostEqual(row["roc_auc"]["effusion"], roc_auc["effusion"])
        self.assertAlmostEqual(row["macro_average_precision"], macro)
        self.assertAlmostEqual(row["average_precision"]["opacity"], average_precision["opacity"])
        self.assertAlmostEqual(row["average_precision"]["effusion"], average_precision["effusion"])

    def test_teacher_updates_once_from_the_final_student(self):
        collection = labeled_collection()
        _student, teacher = initial_student_teacher(collection, seed=0)
        initial_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
        run(self.root, collection, "ema", learning_rate=0.2)
        saved = load_checkpoint(self.root, "ema")
        self.assertNotIn("task_head.weight", saved["teacher"])
        self.assertIn("task_head.weight", saved["student"])
        for key, initial in initial_teacher.items():
            expected = TEACHER_MOMENTUM * initial + (1.0 - TEACHER_MOMENTUM) * saved["student"][key]
            self.assertTrue(torch.allclose(saved["teacher"][key], expected))
        resume = load_resume(self.root, "ema")
        for key, initial in initial_teacher.items():
            expected = TEACHER_MOMENTUM * initial + (1.0 - TEACHER_MOMENTUM) * resume["student"][key]
            self.assertTrue(torch.allclose(resume["teacher"][key], expected))

    def test_training_images_are_not_scored(self):
        validation = labeled_collection().validation
        fit_a = labeled_collection().fit
        fit_b = (
            observation(1.2, {"opacity": 0, "effusion": 0}, augment=1.4),
            observation(1.5, {"opacity": 0, "effusion": 0}, augment=1.7),
        )
        other_validation = (
            observation(0.15, {"opacity": 0, "effusion": 1}),
            observation(0.35, {"opacity": 1, "effusion": 0}),
            observation(0.55, {"opacity": 0, "effusion": 0}),
            observation(0.75, {"opacity": 1, "effusion": 1}),
        )
        first = run(self.root, Collection("fake", ("opacity", "effusion"), fit_a, validation), "fit-a", learning_rate=0.0)
        second = run(self.root, Collection("fake", ("opacity", "effusion"), fit_b, validation), "fit-b", learning_rate=0.0)
        third = run(
            self.root,
            Collection("fake", ("opacity", "effusion"), fit_a, other_validation),
            "other-val",
            learning_rate=0.0,
        )
        self.assertEqual(first[0]["headline_score"], second[0]["headline_score"])
        self.assertEqual(first[0]["roc_auc"], second[0]["roc_auc"])
        self.assertEqual(first[0]["validation_loss"], second[0]["validation_loss"])
        self.assertNotEqual(first[0]["training_loss"], second[0]["training_loss"])
        self.assertNotEqual(first[0]["headline_score"], third[0]["headline_score"])

    def test_checkpoint_is_the_highest_validation_headline(self):
        collection = labeled_collection()
        run(self.root, collection, "tied", epochs=2, learning_rate=0.0)
        tied_checkpoint = load_checkpoint(self.root, "tied")
        tied_resume = load_resume(self.root, "tied")
        self.assertEqual(tied_checkpoint["epoch"], 0)
        self.assertEqual(tied_resume["epoch"], 1)
        self.assertEqual(tied_checkpoint["headline_score"], tied_resume["metrics"][0]["headline_score"])

        metrics = run(self.root, collection, "moving", epochs=2, learning_rate=0.8, seed=1)
        self.assertNotEqual(metrics[0]["headline_score"], metrics[1]["headline_score"])
        best_epoch = max(range(2), key=lambda index: (metrics[index]["headline_score"], -index))
        checkpoint = load_checkpoint(self.root, "moving")
        self.assertEqual(checkpoint["epoch"], best_epoch)
        self.assertAlmostEqual(checkpoint["headline_score"], metrics[best_epoch]["headline_score"])
        if best_epoch == 1:
            for key in checkpoint["student"]:
                self.assertTrue(torch.equal(checkpoint["student"][key], load_resume(self.root, "moving")["student"][key]))
        else:
            moved = False
            resume = load_resume(self.root, "moving")
            for key in checkpoint["student"]:
                moved = moved or not torch.equal(checkpoint["student"][key], resume["student"][key])
            self.assertTrue(moved)

    def test_resume_continues_the_same_experiment_and_keeps_the_metrics_record(self):
        collection = labeled_collection()
        first = run(self.root, collection, "alpha", epochs=1, seed=3, learning_rate=0.15)
        continued = run(self.root, collection, "alpha", epochs=2, seed=3, learning_rate=0.15)
        uninterrupted = run(self.root, collection, "once", epochs=2, seed=3, learning_rate=0.15)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(continued), 2)
        self.assertEqual(continued[0], first[0])
        self.assertEqual(continued[1]["epoch"], 1)
        self.assertEqual(len(uninterrupted), 2)
        for earlier, later in zip(continued, uninterrupted):
            self.assertEqual(earlier["epoch"], later["epoch"])
            self.assertAlmostEqual(earlier["training_loss"], later["training_loss"])
            self.assertAlmostEqual(earlier["headline_score"], later["headline_score"])
        continued_resume = load_resume(self.root, "alpha")
        uninterrupted_resume = load_resume(self.root, "once")
        for key in continued_resume["student"]:
            self.assertTrue(torch.allclose(continued_resume["student"][key], uninterrupted_resume["student"][key]))
        self.assertIn("optimizer", continued_resume)
        self.assertIn("metrics", continued_resume)

    def test_a_second_experiment_name_does_not_load_the_first_checkpoint(self):
        collection = labeled_collection()
        first = run(self.root, collection, "alpha", epochs=1, seed=4, learning_rate=0.3)
        run(self.root, collection, "alpha", epochs=2, seed=4, learning_rate=0.3)
        second = run(self.root, collection, "beta", epochs=1, seed=4, learning_rate=0.3)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["training_loss"], first[0]["training_loss"])
        self.assertEqual(second[0]["headline_score"], first[0]["headline_score"])
        self.assertFalse((self.root / "beta" / "resume.pth").samefile(self.root / "alpha" / "resume.pth"))

    def test_individual_run_list_has_one_collection(self):
        collection = labeled_collection()
        with self.assertRaises(ValueError):
            run_experiment("alpha", "individual", [collection, collection], self.root, epochs=1)
        with self.assertRaises(ValueError):
            run_experiment("alpha", "cyclic", [collection], self.root, epochs=1)
