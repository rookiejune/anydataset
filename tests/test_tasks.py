import unittest

from anydatasets.samples import Sample
from anydatasets.tasks import BatchMeta, ImageClassificationBatch, ImageClassificationTask, Task


class TaskTest(unittest.TestCase):
    def test_task_uses_auto_str_value(self):
        self.assertEqual(Task.IMAGE_CLASSIFICATION.value, "image_classification")

    def test_batch_meta_is_list_aligned(self):
        meta = BatchMeta(dataset_names=["a", "b"], sample_indices=[0, 4])

        self.assertEqual(meta.dataset_names, ["a", "b"])
        self.assertEqual(meta.sample_indices, [0, 4])

    def test_image_classification_task_builds_tensor_batch(self):
        import torch

        task = ImageClassificationTask()
        batch = task.build(
            [
                Sample(data={"image": [[1, 2], [3, 4]], "label": 0}, dataset_name="a", sample_index=0),
                Sample(data={"image": [[5, 6], [7, 8]], "label": 1}, dataset_name="b", sample_index=3),
            ]
        )

        self.assertIsInstance(batch, ImageClassificationBatch)
        self.assertEqual(tuple(batch.images.shape), (2, 1, 2, 2))
        self.assertTrue(torch.equal(batch.labels, torch.tensor([0, 1])))
        self.assertEqual(batch.meta.dataset_names, ["a", "b"])
        self.assertEqual(batch.meta.sample_indices, [0, 3])


if __name__ == "__main__":
    unittest.main()
