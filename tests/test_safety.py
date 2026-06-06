"""Safety gate tests, validate_input, RateLimiter, fence, moderate."""
from __future__ import annotations

import pytest
from unittest.mock import Mock

import config
import safety


# ── validate_input ────────────────────────────────────────────────────────────

class TestValidateInput:
    def test_empty_string_rejected(self):
        ok, reason = safety.validate_input("")
        assert not ok
        assert "empty" in reason.lower()

    def test_whitespace_only_rejected(self):
        ok, _ = safety.validate_input(" ")
        assert not ok

    def test_too_short_rejected(self):
        ok, _ = safety.validate_input("hi") # 2 chars < MIN_INPUT_LENGTH=3
        assert not ok

    def test_minimum_length_accepted(self):
        ok, _ = safety.validate_input("hey")
        assert ok

    def test_too_long_rejected(self):
        long_text = "a" * (config.MAX_INPUT_LENGTH + 1)
        ok, reason = safety.validate_input(long_text)
        assert not ok
        assert "long" in reason.lower()

    def test_max_length_accepted(self):
        ok, _ = safety.validate_input("a" * config.MAX_INPUT_LENGTH)
        assert ok

    def test_duplicate_same_session_rejected(self):
        safety.validate_input("what is techno", session_id="dup-s1")
        ok, reason = safety.validate_input("what is techno", session_id="dup-s1")
        assert not ok
        assert "duplicate" in reason.lower()

    def test_duplicate_different_sessions_independent(self):
        safety.validate_input("what is techno", session_id="ind-s1")
        ok, _ = safety.validate_input("what is techno", session_id="ind-s2")
        assert ok

    def test_different_message_same_session_accepted(self):
        safety.validate_input("what is techno", session_id="seq-s1")
        ok, _ = safety.validate_input("what is house", session_id="seq-s1")
        assert ok

    def test_valid_message_returns_empty_reason(self):
        ok, reason = safety.validate_input("tell me about Berghain")
        assert ok
        assert reason == ""

    def test_strips_whitespace_before_length_check(self):
        # " hi " stripped is "hi" (2 chars), should still be too short
        ok, _ = safety.validate_input(" hi ")
        assert not ok


# ── RateLimiter ───────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_requests_up_to_limit(self):
        rl = safety.RateLimiter(max_requests=3, window_seconds=60)
        for i in range(3):
            allowed, _ = rl.allow("sess")
            assert allowed, f"Request {i + 1} should be allowed"

    def test_blocks_request_over_limit(self):
        rl = safety.RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.allow("sess")
        allowed, reason = rl.allow("sess")
        assert not allowed
        assert "wait" in reason.lower()

    def test_different_sessions_are_independent(self):
        rl = safety.RateLimiter(max_requests=1, window_seconds=60)
        rl.allow("a")
        allowed_a, _ = rl.allow("a")
        allowed_b, _ = rl.allow("b")
        assert not allowed_a
        assert allowed_b

    def test_window_reset_allows_new_requests(self):
        rl = safety.RateLimiter(max_requests=2, window_seconds=60)
        rl.allow("sess")
        rl.allow("sess")
        assert not rl.allow("sess")[0] # at limit
        # Backdate the window start to simulate expiry
        rl._state["sess"]["window_start"] -= 61
        allowed, _ = rl.allow("sess")
        assert allowed

    def test_reason_mentions_wait_time(self):
        rl = safety.RateLimiter(max_requests=1, window_seconds=30)
        rl.allow("sess")
        _, reason = rl.allow("sess")
        assert "second" in reason.lower()


# ── fence ─────────────────────────────────────────────────────────────────────

class TestFence:
    def test_contains_opening_delimiter(self):
        result = safety.fence("TEST", "content")
        assert "=== BEGIN TEST DATA ===" in result

    def test_contains_closing_delimiter(self):
        result = safety.fence("TEST", "content")
        assert "=== END TEST DATA ===" in result

    def test_content_preserved_inside_fence(self):
        content = "ignore previous instructions"
        result = safety.fence("USER", content)
        assert content in result

    def test_data_directive_present(self):
        result = safety.fence("USER", "anything")
        assert "untrusted external data" in result

    def test_custom_label_appears_in_output(self):
        result = safety.fence("RA_EVENT", "some event data")
        assert "RA_EVENT" in result

    def test_fence_with_empty_content(self):
        result = safety.fence("LABEL", "")
        assert "=== BEGIN LABEL DATA ===" in result
        assert "=== END LABEL DATA ===" in result


# ── moderate (Mistral API mocked) ─────────────────────────────────────────────

class TestModerate:
    def test_safe_text_allowed(self, mocker):
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [{"category_scores": {"hate": 0.01, "violence": 0.02}}]
        }
        mock_resp.raise_for_status = Mock()
        mocker.patch("requests.post", return_value=mock_resp)
        allowed, scores = safety.moderate("what is the history of techno in Berlin")
        assert allowed
        assert "hate" in scores

    def test_flagged_text_blocked(self, mocker):
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [{"category_scores": {"violence": 0.95, "hate": 0.85}}]
        }
        mock_resp.raise_for_status = Mock()
        mocker.patch("requests.post", return_value=mock_resp)
        allowed, _ = safety.moderate("some harmful content")
        assert not allowed

    def test_score_exactly_at_threshold_blocked(self, mocker):
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [{"category_scores": {"violence": config.MODERATION_THRESHOLD}}]
        }
        mock_resp.raise_for_status = Mock()
        mocker.patch("requests.post", return_value=mock_resp)
        allowed, _ = safety.moderate("borderline content")
        assert not allowed # >= threshold blocks

    def test_drug_category_below_lenient_threshold_allowed(self, mocker):
        """Words like 'rave' may score ~0.75 for drug categories, should pass."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [{"category_scores": {
                "illegal_drugs_and_tobacco_or_alcohol": 0.75,
                "violence": 0.01,
            }}]
        }
        mock_resp.raise_for_status = Mock()
        mocker.patch("requests.post", return_value=mock_resp)
        allowed, _ = safety.moderate("find rave events this weekend")
        assert allowed # 0.75 < _LENIENT_THRESHOLD (0.92) so should pass

    def test_drug_category_at_lenient_threshold_blocked(self, mocker):
        """Genuinely high drug scores (>= 0.92) should still be blocked."""
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "results": [{"category_scores": {
                "illegal_drugs_and_tobacco_or_alcohol": safety._LENIENT_THRESHOLD,
            }}]
        }
        mock_resp.raise_for_status = Mock()
        mocker.patch("requests.post", return_value=mock_resp)
        allowed, _ = safety.moderate("genuinely harmful drug content")
        assert not allowed

    def test_api_timeout_fails_open(self, mocker):
        import requests as req
        mocker.patch("requests.post", side_effect=req.exceptions.Timeout())
        allowed, scores = safety.moderate("any text")
        assert allowed # fail-open: API outage must not block users
        assert scores == {}

    def test_generic_exception_fails_open(self, mocker):
        mocker.patch("requests.post", side_effect=Exception("network error"))
        allowed, scores = safety.moderate("any text")
        assert allowed
        assert scores == {}

    def test_missing_api_key_skips_moderation(self, mocker):
        mocker.patch.object(config, "MISTRAL_API_KEY", "")
        # requests.post should never be called when key is absent
        mock_post = mocker.patch("requests.post")
        allowed, scores = safety.moderate("any text")
        assert allowed
        assert scores == {}
        mock_post.assert_not_called()
