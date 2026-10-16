from rail_rag.ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_per_client_minute_window_slides():
    clock = Clock()
    rl = RateLimiter(per_client_per_minute=2, per_day=100, clock=clock)
    assert rl.check("a") is None and rl.check("a") is None
    assert "minute" in rl.check("a")
    assert rl.check("b") is None  # other clients unaffected
    clock.t = 60
    assert rl.check("a") is None


def test_daily_budget_is_global():
    clock = Clock()
    rl = RateLimiter(per_client_per_minute=100, per_day=3, clock=clock)
    assert [rl.check(c) for c in "abc"] == [None, None, None]
    assert "daily" in rl.check("d")
    clock.t = 86_400
    assert rl.check("d") is None


def test_rejected_requests_do_not_consume_budget():
    rl = RateLimiter(per_client_per_minute=1, per_day=2, clock=Clock())
    rl.check("a")
    assert rl.check("a") is not None
    assert rl.check("b") is None  # 'a' being rejected did not use up the daily budget
