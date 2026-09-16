"""Tests for RetryQueue: attempt tracking, exponential backoff, give-up."""

from pathlib import Path

from obsidian_rag.retry_queue import RetryQueue

A = Path("/vault/a.md")
B = Path("/vault/b.md")


class TestAdd:
    def test_new_path_is_due_immediately(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(A, now=100.0)
        assert not queue.is_empty()
        assert queue.pop_due(now=100.0) == [A]

    def test_add_is_idempotent(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(A, now=100.0)
        queue.add(A, now=100.0)
        assert len(queue) == 1

    def test_add_does_not_reset_attempts(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(A, now=0.0)
        assert queue.pop_due(now=0.0) == [A]
        assert queue.mark_failed(A, now=0.0) is True
        assert queue.attempts(A) == 1
        queue.add(A, now=0.0)
        assert queue.attempts(A) == 1


class TestPopDue:
    def test_returns_only_due_items_and_marks_them_in_flight(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(A, now=0.0)
        queue.add(B, now=0.0)
        queue.pop_due(now=0.0)
        queue.mark_failed(A, now=0.0)  # A due again at 60
        queue.add(B, now=0.0)  # B due immediately
        assert queue.pop_due(now=10.0) == [B]
        assert queue.pop_due(now=10.0) == []  # B is in flight, not re-offered
        assert queue.attempts(A) == 1  # A still tracked with its history
        assert len(queue) == 2

    def test_is_a_snapshot_not_a_live_view(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(A, now=0.0)
        due = queue.pop_due(now=0.0)
        queue.add(B, now=0.0)
        assert due == [A]


class TestMarkFailed:
    def test_exponential_backoff_schedule(self):
        queue = RetryQueue(max_retries=5, base_delay=60.0)
        queue.add(A, now=0.0)
        queue.pop_due(now=0.0)

        queue.mark_failed(A, now=0.0)
        assert queue.pop_due(now=59.0) == []
        assert queue.pop_due(now=60.0) == [A]

        queue.mark_failed(A, now=60.0)
        assert queue.pop_due(now=179.0) == []
        assert queue.pop_due(now=180.0) == [A]

        queue.mark_failed(A, now=180.0)
        assert queue.pop_due(now=419.0) == []
        assert queue.pop_due(now=420.0) == [A]

    def test_gives_up_after_max_retries(self):
        given_up: list[Path] = []
        queue = RetryQueue(max_retries=3, base_delay=1.0, on_give_up=given_up.append)
        queue.add(A, now=0.0)
        now = 0.0
        results = []
        for _ in range(3):
            assert queue.pop_due(now=now) == [A]
            results.append(queue.mark_failed(A, now=now))
            now += 1000.0

        assert results == [True, True, False]
        assert queue.is_empty()
        assert given_up == [A]
        assert queue.attempts(A) == 0

    def test_mark_failed_on_unknown_path_counts_first_attempt(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        assert queue.mark_failed(A, now=0.0) is True
        assert queue.attempts(A) == 1
        assert len(queue) == 1


class TestDiscard:
    def test_discard_removes_item_and_attempts(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.add(A, now=0.0)
        queue.pop_due(now=0.0)
        queue.mark_failed(A, now=0.0)
        queue.discard(A)
        assert queue.is_empty()
        assert queue.attempts(A) == 0

    def test_discard_unknown_path_is_noop(self):
        queue = RetryQueue(max_retries=3, base_delay=60.0)
        queue.discard(A)
        assert queue.is_empty()
