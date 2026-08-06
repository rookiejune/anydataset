from __future__ import annotations

import threading
import unittest

from anydataset.dataset.coverage import (
    CoverageAbortedError,
    CoverageCoordinator,
)


class CoverageCoordinatorTest(unittest.TestCase):
    def test_initial_completion_and_validation(self):
        coordinator = CoverageCoordinator(3, completed=(0, 2))

        self.assertEqual(coordinator.expected, 3)
        self.assertEqual(coordinator.completed_count, 2)
        self.assertFalse(coordinator.complete)
        self.assertFalse(coordinator.claim(0).owner)
        self.assertIsNone(coordinator.claim(0).wait())

        with self.assertRaises(ValueError):
            CoverageCoordinator(-1)
        with self.assertRaises(IndexError):
            CoverageCoordinator(2, completed=(2,))
        with self.assertRaises(IndexError):
            coordinator.claim(-1)
        with self.assertRaises(TypeError):
            coordinator.claim(True)
        with self.assertRaises(TypeError):
            coordinator.claim(1, require_value=1)  # type: ignore[arg-type]

    def test_same_index_waiter_receives_owner_value(self):
        coordinator = CoverageCoordinator(1)
        owner = coordinator.claim(0, require_value=True)
        waiter = coordinator.claim(0, require_value=True)
        value = object()

        self.assertTrue(owner.owner)
        self.assertFalse(waiter.owner)
        owner.complete(value)

        self.assertIs(owner.wait(), value)
        self.assertIs(waiter.wait(), value)
        self.assertIs(waiter.wait(), value)
        self.assertTrue(coordinator.complete)

    def test_completed_index_recomputes_only_when_value_is_required(self):
        coordinator = CoverageCoordinator(1, completed=(0,))

        covered = coordinator.claim(0)
        recompute = coordinator.claim(0, require_value=True)
        waiter = coordinator.claim(0, require_value=True)

        self.assertFalse(covered.owner)
        self.assertTrue(recompute.owner)
        self.assertFalse(waiter.owner)
        recompute.complete("fresh")
        self.assertEqual(waiter.wait(), "fresh")

        next_recompute = coordinator.claim(0, require_value=True)
        self.assertTrue(next_recompute.owner)
        next_recompute.complete("newer")

    def test_value_is_retained_only_for_active_value_waiters(self):
        coordinator = CoverageCoordinator(1)
        owner = coordinator.claim(0)
        first = coordinator.claim(0, require_value=True)
        second = coordinator.claim(0, require_value=True)
        owner.complete([1, 2, 3])

        self.assertEqual(first.wait(), [1, 2, 3])
        late = coordinator.claim(0, require_value=True)
        self.assertFalse(late.owner)
        self.assertEqual(second.wait(), [1, 2, 3])
        self.assertEqual(late.wait(), [1, 2, 3])

        recompute = coordinator.claim(0, require_value=True)
        self.assertTrue(recompute.owner)
        recompute.complete(None)

    def test_failure_is_global_and_sticky(self):
        coordinator = CoverageCoordinator(2)
        owner = coordinator.claim(0)
        waiter = coordinator.claim(0, require_value=True)
        failure = RuntimeError("provider failed")

        owner.fail(failure)

        for action in (
            waiter.wait,
            lambda: coordinator.claim(1),
            coordinator.wait_complete,
            coordinator.close,
        ):
            with self.assertRaises(RuntimeError) as raised:
                action()
            self.assertIs(raised.exception, failure)

    def test_full_sweep_skips_completed_and_inflight_indexes(self):
        coordinator = CoverageCoordinator(5, completed=(0, 4))
        foreground = coordinator.claim(2)

        sweep = list(coordinator.full_sweep())

        self.assertEqual([lease.index for lease in sweep], [1, 3])
        self.assertTrue(all(lease.owner for lease in sweep))
        foreground.complete(None)
        for lease in sweep:
            lease.complete(None)
        coordinator.wait_complete()
        self.assertEqual(coordinator.completed_count, 5)

    def test_wait_complete_does_not_close_coordinator(self):
        coordinator = CoverageCoordinator(2)
        first = coordinator.claim(0)
        second = coordinator.claim(1)
        returned = threading.Event()

        thread = threading.Thread(
            target=lambda: (coordinator.wait_complete(), returned.set())
        )
        thread.start()
        self.assertFalse(returned.wait(timeout=0.05))
        first.complete(None)
        self.assertFalse(returned.wait(timeout=0.05))
        second.complete(None)
        self.assertTrue(returned.wait(timeout=5))
        thread.join(timeout=5)

        recompute = coordinator.claim(0, require_value=True)
        self.assertTrue(recompute.owner)
        recompute.complete("value")

    def test_close_blocks_claims_and_waits_for_current_owner(self):
        coordinator = CoverageCoordinator(1)
        owner = coordinator.claim(0)
        returned = threading.Event()
        thread = threading.Thread(target=lambda: (coordinator.close(), returned.set()))
        thread.start()

        self.assertFalse(returned.wait(timeout=0.05))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            coordinator.claim(0)
        owner.complete(None)
        self.assertTrue(returned.wait(timeout=5))
        thread.join(timeout=5)

    def test_abort_wakes_waiters_and_invalidates_owner(self):
        coordinator = CoverageCoordinator(1)
        owner = coordinator.claim(0)
        waiter = coordinator.claim(0, require_value=True)
        returned = threading.Event()
        errors = []

        def wait() -> None:
            try:
                waiter.wait()
            except BaseException as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=wait)
        thread.start()
        coordinator.abort()

        self.assertTrue(returned.wait(timeout=5))
        thread.join(timeout=5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CoverageAbortedError)
        with self.assertRaises(CoverageAbortedError):
            coordinator.claim(0)
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            owner.complete(None)

    def test_batch_claim_preserves_order_and_reports_owners(self):
        coordinator = CoverageCoordinator(3, completed=(0,))
        batch = coordinator.claim_batch((0, 1, 1, 2))

        self.assertEqual([lease.index for lease in batch], [0, 1, 1, 2])
        self.assertEqual([lease.index for lease in batch.owners], [1, 2])
        for owner in batch.owners:
            owner.complete(None)
        self.assertEqual(batch.wait(), (None, None, None, None))
        coordinator.wait_complete()


if __name__ == "__main__":
    unittest.main()
