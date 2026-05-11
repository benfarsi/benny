from datetime import datetime
from config import STARTING_BALANCE, TRADE_SIZE

FEE = 0.001  # 0.1% Binance taker fee on each side


class PaperTrader:
    def __init__(self):
        self.usdt = STARTING_BALANCE
        self.btc = 0.0
        self.trades: list[dict] = []

    def buy(self, price: float, spend_usdt: float | None = None) -> bool:
        if self.usdt <= 0:
            return False
        spend = min(spend_usdt, self.usdt) if spend_usdt is not None else self.usdt * TRADE_SIZE
        if spend <= 0:
            return False
        self.btc += spend / price * (1 - FEE)
        self.usdt -= spend
        trade = {
            "type": "BUY",
            "price": price,
            "btc": self.btc,
            "usdt": self.usdt,
            "time": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self.trades.append(trade)
        return True

    def sell(self, price: float) -> bool:
        if self.btc <= 0:
            return False
        self.usdt += self.btc * price * (1 - FEE)
        self.btc = 0.0
        trade = {
            "type": "SELL",
            "price": price,
            "btc": self.btc,
            "usdt": self.usdt,
            "time": datetime.utcnow().isoformat(timespec="seconds"),
        }
        self.trades.append(trade)
        return True

    def portfolio_value(self, price: float) -> float:
        return self.usdt + self.btc * price

    def print_status(self, price: float, signal: str, rsi: float, ema: float) -> None:
        total = self.portfolio_value(price)
        pnl_pct = (total - STARTING_BALANCE) / STARTING_BALANCE * 100
        last = self.trades[-1] if self.trades else None
        last_str = (
            f"{last['type']} @ ${last['price']:,.2f} ({last['time']} UTC)"
            if last
            else "none"
        )

        signal_colors = {"BUY": "\033[92m", "SELL": "\033[91m", "HOLD": "\033[93m"}
        pnl_color = "\033[92m" if pnl_pct >= 0 else "\033[91m"
        reset = "\033[0m"
        color = signal_colors.get(signal, reset)

        print("=" * 52)
        print(f"  {'BTC/USDT PAPER TRADER':^48}")
        print("=" * 52)
        print(f"  Time       : {datetime.utcnow().isoformat(timespec='seconds')} UTC")
        print(f"  Price      : ${price:>12,.2f}")
        print(f"  EMA({str(20):>2})    : ${ema:>12,.2f}")
        print(f"  RSI({str(14):>2})    : {rsi:>12.2f}")
        print(f"  Signal     : {color}{signal:>12}{reset}")
        print("-" * 52)
        print(f"  USDT       : ${self.usdt:>12,.2f}")
        print(f"  BTC held   : {self.btc:>13.6f}")
        print(f"  Portfolio  : ${total:>12,.2f}")
        print(f"  P&L        : {pnl_color}{pnl_pct:>+11.2f}%{reset}")
        print(f"  Last trade : {last_str}")
        print("=" * 52)
