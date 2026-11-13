from monitoring.alerts import Alerts


def test_alerts_fire_on_drawdown():
    alerts = Alerts(slack_webhook="https://hooks.slack.com/test")
    alerts.on_drawdown(drawdown=0.05)  # 5% drawdown
    assert alerts.was_fired


def test_alerts_fire_on_daily_loss():
    alerts = Alerts()  # default daily_loss_limit = 1.5%
    alerts.on_daily_loss(daily_loss=0.016)  # 1.6% > 1.5%
    assert alerts.was_fired
