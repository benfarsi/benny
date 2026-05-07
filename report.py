import csv
import os
from collections import defaultdict

_HERE      = os.path.dirname(os.path.abspath(__file__))
_TRADE_LOG = os.path.join(_HERE, "data", "trades.csv")


def _load_trades():
    if not os.path.exists(_TRADE_LOG):
        return []
    with open(_TRADE_LOG, newline="") as f:
        return list(csv.DictReader(f))


def _pair_trades(rows):
    by_sym = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(r)

    pairs = []
    for sym, trades in by_sym.items():
        open_trade = None
        for t in trades:
            if t["side"] == "BUY":
                open_trade = t
            elif t["side"] == "SELL" and open_trade:
                buy_p  = float(open_trade["price"])
                sell_p = float(t["price"])
                pnl    = (sell_p - buy_p) / buy_p * 100
                ml_p   = float(open_trade["ml_proba"]) if open_trade.get("ml_proba") else None
                pairs.append({
                    "symbol":     sym,
                    "buy_time":   open_trade["time"],
                    "sell_time":  t["time"],
                    "buy_price":  buy_p,
                    "sell_price": sell_p,
                    "pnl_pct":    round(pnl, 4),
                    "trigger":    t.get("trigger", "ml"),
                    "ml_proba":   ml_p,
                    "won":        pnl > 0,
                })
                open_trade = None
    pairs.sort(key=lambda x: x["buy_time"])
    return pairs


def generate_report():
    rows  = _load_trades()
    pairs = _pair_trades(rows)

    total_buys   = sum(1 for r in rows if r["side"] == "BUY")
    total_sells  = sum(1 for r in rows if r["side"] == "SELL")
    open_positions = total_buys - total_sells

    if not pairs:
        return {
            "summary": {
                "total_trades":     len(rows),
                "completed_trades": 0,
                "open_positions":   open_positions,
                "win_rate":         None,
                "profit_factor":    None,
                "avg_win":          None,
                "avg_loss":         None,
                "expectancy":       None,
                "best_symbol":      None,
                "worst_symbol":     None,
            },
            "by_symbol":         [],
            "ml_effectiveness":  None,
            "trigger_breakdown": {},
            "recommendations":   ["Not enough completed trades yet — let the bot run a few days and check back."],
            "recent_pairs":      [],
        }

    winners = [p for p in pairs if p["won"]]
    losers  = [p for p in pairs if not p["won"]]

    win_rate      = len(winners) / len(pairs) * 100
    avg_win       = sum(p["pnl_pct"] for p in winners) / len(winners) if winners else 0.0
    avg_loss      = sum(p["pnl_pct"] for p in losers)  / len(losers)  if losers  else 0.0
    gross_win     = sum(p["pnl_pct"] for p in winners)
    gross_loss    = abs(sum(p["pnl_pct"] for p in losers))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else None
    expectancy    = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # per-symbol stats
    by_sym = defaultdict(list)
    for p in pairs:
        by_sym[p["symbol"]].append(p)

    sym_stats = []
    for sym, ps in by_sym.items():
        w = [p for p in ps if p["won"]]
        sym_stats.append({
            "symbol":      sym,
            "trades":      len(ps),
            "win_rate":    round(len(w) / len(ps) * 100, 1),
            "avg_pnl":     round(sum(p["pnl_pct"] for p in ps) / len(ps), 3),
            "best_trade":  round(max(p["pnl_pct"] for p in ps), 3),
            "worst_trade": round(min(p["pnl_pct"] for p in ps), 3),
            "total_pnl":   round(sum(p["pnl_pct"] for p in ps), 3),
        })
    sym_stats.sort(key=lambda x: x["total_pnl"], reverse=True)

    best_sym  = sym_stats[0]["symbol"]  if sym_stats else None
    worst_sym = sym_stats[-1]["symbol"] if len(sym_stats) > 1 else None

    # ML effectiveness
    ml_pairs = [p for p in pairs if p["ml_proba"] is not None]
    ml_eff   = None
    if ml_pairs:
        ml_win  = [p["ml_proba"] for p in ml_pairs if p["won"]]
        ml_loss = [p["ml_proba"] for p in ml_pairs if not p["won"]]
        n_high  = sum(1 for p in ml_pairs if p["ml_proba"] > 0.45)
        n_high_won = sum(1 for p in ml_pairs if p["ml_proba"] > 0.45 and p["won"])
        ml_eff = {
            "avg_proba_winners":  round(sum(ml_win)  / len(ml_win),  4) if ml_win  else None,
            "avg_proba_losers":   round(sum(ml_loss) / len(ml_loss), 4) if ml_loss else None,
            "n_high_conf":        n_high,
            "win_rate_high_conf": round(n_high_won / max(1, n_high) * 100, 1),
        }

    # trigger breakdown
    trig = defaultdict(lambda: {"count": 0, "wins": 0})
    for p in pairs:
        t = p["trigger"]
        trig[t]["count"] += 1
        if p["won"]:
            trig[t]["wins"] += 1
    trigger_breakdown = {
        k: {"count": v["count"], "win_rate": round(v["wins"] / v["count"] * 100, 1)}
        for k, v in trig.items()
    }

    recs = _make_recommendations(
        win_rate, avg_win, avg_loss, profit_factor,
        expectancy, sym_stats, ml_eff, trigger_breakdown, pairs,
    )

    return {
        "summary": {
            "total_trades":     len(rows),
            "completed_trades": len(pairs),
            "open_positions":   open_positions,
            "win_rate":         round(win_rate, 1),
            "profit_factor":    round(profit_factor, 2) if profit_factor is not None else None,
            "avg_win":          round(avg_win, 3),
            "avg_loss":         round(avg_loss, 3),
            "expectancy":       round(expectancy, 3),
            "best_symbol":      best_sym,
            "worst_symbol":     worst_sym,
        },
        "by_symbol":         sym_stats,
        "ml_effectiveness":  ml_eff,
        "trigger_breakdown": trigger_breakdown,
        "recommendations":   recs,
        "recent_pairs":      list(reversed(pairs[-10:])),
    }


def _make_recommendations(win_rate, avg_win, avg_loss, profit_factor,
                           expectancy, sym_stats, ml_eff, triggers, pairs):
    recs = []

    if len(pairs) < 5:
        recs.append("need_data: Run at least 5 completed round-trip trades before drawing conclusions.")
        return recs

    # overall edge
    if expectancy > 0.15:
        recs.append(f"positive: Strategy has solid positive expectancy ({expectancy:+.2f}% per trade). Keep running as-is.")
    elif expectancy > 0:
        recs.append(f"warn: Marginally positive expectancy ({expectancy:+.2f}%). Watch for fees eating the edge on small wins.")
    else:
        recs.append(f"negative: Negative expectancy ({expectancy:+.2f}%). Losing money on average — retrain the model or raise BUY_CONF threshold.")

    # win rate vs payoff ratio
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    if win_rate < 40 and payoff < 1.5:
        recs.append("warn: Low win rate AND small avg win vs avg loss. Raise BUY_CONF to only take your strongest signals.")
    if win_rate > 65 and profit_factor is not None and profit_factor < 1.3:
        recs.append("warn: High win rate but winners are too small vs losers. Consider raising TAKE_PROFIT from 3% to 4-5%.")
    if payoff >= 2.0 and win_rate >= 35:
        recs.append(f"positive: Good payoff ratio ({payoff:.1f}x) — winners are much larger than losers. This is sustainable.")

    # stop loss / take profit balance
    sl = triggers.get("stop_loss",   {})
    tp = triggers.get("take_profit", {})
    if sl.get("count", 0) > 0 and tp.get("count", 0) > 0:
        if sl["count"] > tp["count"] * 2:
            recs.append("warn: Stop loss firing 2x more than take profit. Price moving against entries too often — consider raising BUY_CONF or widening STOP_LOSS slightly.")
        if tp["count"] > sl["count"] * 3:
            recs.append("tip: Take profit firing much more than stop loss — you may be cutting winners too early. Try raising TAKE_PROFIT to 4-5%.")

    # ML signal quality
    if ml_eff:
        wp = ml_eff.get("avg_proba_winners")
        lp = ml_eff.get("avg_proba_losers")
        if wp is not None and lp is not None:
            diff = wp - lp
            if diff < 0.02:
                recs.append("warn: ML proba barely differs between winners and losers — model may not be predictive. Retrain with more recent data.")
            elif diff > 0.05:
                recs.append(f"positive: ML model is doing real work — winners avg {wp*100:.1f}% proba vs losers {lp*100:.1f}%.")
        if ml_eff["n_high_conf"] >= 3:
            recs.append(f"tip: High-confidence signals (>45% proba) win {ml_eff['win_rate_high_conf']}% of the time ({ml_eff['n_high_conf']} signals). Consider raising BUY_CONF to 0.45 to only trade these.")

    # per-symbol
    if len(sym_stats) >= 2:
        worst = sym_stats[-1]
        best  = sym_stats[0]
        if worst["avg_pnl"] < -0.3 and worst["trades"] >= 3:
            recs.append(f"negative: {worst['symbol']} is consistently losing (avg {worst['avg_pnl']:+.2f}% per trade, {worst['trades']} trades). Consider removing it from config.")
        if best["win_rate"] >= 60 and best["trades"] >= 3:
            recs.append(f"positive: {best['symbol']} is your best performer ({best['win_rate']}% win rate, avg {best['avg_pnl']:+.2f}%). Could allocate more starting capital here.")

    # standing advice
    recs.append("tip: Retrain the model weekly — it will improve as it accumulates live BTC price data.")
    recs.append("tip: Adding Reddit/Twitter sentiment as a signal (fear spikes = contrarian buy) could give the model an edge the pure price data misses.")

    return recs
