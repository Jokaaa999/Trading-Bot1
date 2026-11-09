"""
Paper-Trading-Bot V2 – nur Demo, es wird kein echtes Geld gehandelt.

Strategie: Doppelboden / Doppeltop
Kombinationen: XRP 1h, ETH 1h, ETH 4h, XRP 4h, SOL 4h, BNB 4h (Beobachtung)
Risiko: 2 % des Demo-Kontos pro Trade, max. 3 Trades gleichzeitig
Läuft automatisch jede Stunde über GitHub Actions und meldet sich per Telegram.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ===== DATEIEN =====
ORDNER = "daten"
LOGDATEI = os.path.join(ORDNER, "paper_trades.csv")
STATUSDATEI = os.path.join(ORDNER, "status.json")
os.makedirs(ORDNER, exist_ok=True)

# ===== TELEGRAM (kommt aus den GitHub-Secrets) =====
TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TEST = os.environ.get("TEST_NACHRICHT", "0") == "1"

# ===== EINSTELLUNGEN =====
STARTKAPITAL, RISIKO, MAX_OFFEN = 2000, 0.02, 3
SL_ATR, MM = 2.0, 0.005  # Stop = 2 x ATR, Wartungsmarge für Liquidation ca. 0,5 %
GEBUEHR_SEITE = {"ETH": 0.0005, "BNB": 0.0005, "XRP": 0.0, "SOL": 0.0}
KOMBIS = [("XRP", "1h"), ("ETH", "1h"), ("ETH", "4h"), ("XRP", "4h"), ("SOL", "4h"), ("BNB", "4h")]
BEOBACHTUNG = {("BNB", "4h")}
REGELN = {
    "1h": dict(max_hebel=20, max_bars=48, we_schliessen=True, gegensignal=False, dauer=pd.Timedelta("1h")),
    "4h": dict(max_hebel=10, max_bars=None, we_schliessen=False, gegensignal=True, dauer=pd.Timedelta("4h")),
}
SPALTEN = ["ID", "Coin", "Chart", "Richtung", "Signal_Kerze", "Einstieg_Zeit", "Einstieg", "SL", "TP",
           "Stop_pct", "Risiko_$", "Position_$", "Hebel", "Margin_$", "Liq_Preis",
           "Status", "Ausstieg_Zeit", "Ausstieg", "R", "Ergebnis_$"]


# ===== HILFSFUNKTIONEN =====
def telegram(text):
    print(text)
    if not TOKEN or not CHAT_ID:
        print("Telegram: Secrets fehlen! TOKEN gesetzt:", bool(TOKEN), "| CHAT_ID gesetzt:", bool(CHAT_ID))
        return
    try:
        antwort = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                                data={"chat_id": CHAT_ID, "text": text}, timeout=20)
        print("Telegram:", antwort.status_code, antwort.text[:200])
    except Exception as e:
        print("Telegram-Fehler:", e)



def lade_status():
    if os.path.exists(STATUSDATEI):
        with open(STATUSDATEI, encoding="utf-8") as f:
            return json.load(f)
    return {}


def speichere_status(status):
    with open(STATUSDATEI, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def atr_von(df):
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean().values


def doppel_signale(H, L, C, atr, tol_atr=0.75, k=5):
    n = len(C)
    ist_hoch, ist_tief = np.zeros(n, bool), np.zeros(n, bool)
    for t in range(k, n - k):
        ist_hoch[t] = H[t] == H[t - k:t + k + 1].max()
        ist_tief[t] = L[t] == L[t - k:t + k + 1].min()
    sig = np.zeros(n, int)
    for i in range(80, n):
        if np.isnan(atr[i]):
            continue
        tol = tol_atr * atr[i] / C[i]
        bis = i - k
        hochs = np.where(ist_hoch[i - 80:bis + 1])[0] + (i - 80)
        tiefs = np.where(ist_tief[i - 80:bis + 1])[0] + (i - 80)
        if len(tiefs) >= 2:
            t1, t2 = tiefs[-2], tiefs[-1]
            if t2 - t1 >= 5 and abs(L[t1] - L[t2]) / L[t1] < tol:
                if C[i] > H[t1:t2 + 1].max() >= C[i - 1]:
                    sig[i] = 1
        if len(hochs) >= 2:
            h1, h2 = hochs[-2], hochs[-1]
            if h2 - h1 >= 5 and abs(H[h1] - H[h2]) / H[h1] < tol:
                if C[i] < L[h1:h2 + 1].min() <= C[i - 1]:
                    sig[i] = -1
    return sig


def lade(coin, versuche=3):
    x = pd.DataFrame()
    for _ in range(versuche):
        try:
            x = yf.download(f"{coin}-USD", period="120d", interval="1h",
                            auto_adjust=True, progress=False, threads=False)
            if len(x):
                break
        except Exception as e:
            print(f"Download-Fehler {coin}: {e}")
        time.sleep(10)
    if not len(x):
        raise RuntimeError(f"Keine Kursdaten für {coin} erhalten")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x[["Open", "High", "Low", "Close", "Volume"]]
    x.index = x.index.tz_convert("UTC")
    jetzt = pd.Timestamp.now(tz="UTC")
    h1 = x[x.index + pd.Timedelta(hours=1) <= jetzt]  # nur fertige Kerzen
    h4 = x.resample("4h", origin="epoch").agg({"Open": "first", "High": "max", "Low": "min",
                                               "Close": "last", "Volume": "sum"}).dropna()
    h4 = h4[h4.index + pd.Timedelta(hours=4) <= jetzt]
    h1.index = h1.index.tz_convert("Europe/Berlin")
    h4.index = h4.index.tz_convert("Europe/Berlin")
    return {"1h": h1, "4h": h4}


def kontostand(log, ohne=None):
    f = log[log["Status"] != "offen"]
    if ohne:
        maske = np.array([(c, t) not in ohne for c, t in zip(f["Coin"], f["Chart"])], dtype=bool)
        f = f.loc[maske]
    return STARTKAPITAL + pd.to_numeric(f["Ergebnis_$"], errors="coerce").fillna(0).sum()


def auswerten(log, markt, meldungen):
    for idx, z in log[log["Status"] == "offen"].iterrows():
        df, atr, sig = markt[(z["Coin"], z["Chart"])]
        r = REGELN[z["Chart"]]
        s = 1 if z["Richtung"] == "LONG" else -1
        ein, sl, tp, sl_d = float(z["Einstieg"]), float(z["SL"]), float(z["TP"]), float(z["Stop_pct"]) / 100
        pos = np.where(df.index >= pd.Timestamp(z["Einstieg_Zeit"]))[0]
        H, L, C = df["High"].values, df["Low"].values, df["Close"].values
        we = df.index.dayofweek.values >= 5
        erg = status = None
        for n_, j in enumerate(pos):
            if r["we_schliessen"] and we[j]:
                kurs, erg, status, aus = C[j - 1], s * (C[j - 1] - ein) / ein, "Wochenende", df.index[j]
                break
            if (s == 1 and L[j] <= sl) or (s == -1 and H[j] >= sl):
                kurs, erg, status, aus = sl, -sl_d, "Stop", df.index[j] + r["dauer"]
                break
            if (s == 1 and H[j] >= tp) or (s == -1 and L[j] <= tp):
                kurs, erg, status, aus = tp, 3 * sl_d, "Ziel", df.index[j] + r["dauer"]
                break
            if r["gegensignal"] and sig[j] == -s:
                kurs, erg, status, aus = C[j], s * (C[j] - ein) / ein, "Gegensignal", df.index[j] + r["dauer"]
                break
            if r["max_bars"] and n_ + 1 >= r["max_bars"]:
                kurs, erg, status, aus = C[j], s * (C[j] - ein) / ein, "Zeitlimit", df.index[j] + r["dauer"]
                break
        if status:
            R = (erg - 2 * GEBUEHR_SEITE[z["Coin"]]) / sl_d
            dollar = float(z["Risiko_$"]) * R
            log.loc[idx, ["Status", "Ausstieg_Zeit", "Ausstieg", "R", "Ergebnis_$"]] = \
                [status, aus.isoformat(), kurs, round(R, 3), round(dollar, 2)]
            symbol = "✅" if R > 0 else "❌"
            meldungen.append(f"{symbol} GESCHLOSSEN: {z['Coin']} {z['Chart']} {z['Richtung']}\n"
                             f"Grund: {status}\n"
                             f"Ergebnis: {R:+.2f} R ({dollar:+.2f} $)")
    return log


def zusammenfassung(log, markt, titel):
    konto, konto_ohne = kontostand(log), kontostand(log, ohne=BEOBACHTUNG)
    zeilen = [titel,
              f"💰 Demo-Konto: {konto:,.2f} $ ({(konto / STARTKAPITAL - 1) * 100:+.2f} %)",
              f"   ohne BNB: {konto_ohne:,.2f} $ ({(konto_ohne / STARTKAPITAL - 1) * 100:+.2f} %)"]
    fertig = log[log["Status"] != "offen"]
    if len(fertig):
        zeilen.append("")
        for (coin, tf), g in fertig.groupby(["Coin", "Chart"]):
            r = pd.to_numeric(g["R"], errors="coerce")
            d = pd.to_numeric(g["Ergebnis_$"], errors="coerce")
            zeilen.append(f"{coin} {tf}: {len(g)} Trades | Treffer {(r > 0).mean() * 100:.0f} % | "
                          f"Ø {r.mean():+.2f} R | {d.sum():+.2f} $")
    else:
        zeilen.append("Noch keine abgeschlossenen Trades.")
    offen = log[log["Status"] == "offen"]
    zeilen.append(f"\n📂 Offene Trades: {len(offen)}")
    for _, z in offen.iterrows():
        kurs = markt[(z["Coin"], z["Chart"])][0]["Close"].iloc[-1]
        s = 1 if z["Richtung"] == "LONG" else -1
        r_akt = s * (kurs - float(z["Einstieg"])) / float(z["Einstieg"]) / (float(z["Stop_pct"]) / 100)
        zeilen.append(f"   {z['Coin']} {z['Chart']} {z['Richtung']} | aktuell {r_akt:+.2f} R")
    return "\n".join(zeilen)


# ===== HAUPTPROGRAMM =====
def main():
    jetzt = pd.Timestamp.now(tz="Europe/Berlin")
    print(f"Bot-Lauf {jetzt:%d.%m.%Y %H:%M} Uhr")
    status = lade_status()
    log = pd.read_csv(LOGDATEI) if os.path.exists(LOGDATEI) else pd.DataFrame(columns=SPALTEN)

    # Marktdaten und Signale
    markt = {}
    for coin in sorted({c for c, _ in KOMBIS}):
        d = lade(coin)
        for tf in ("1h", "4h"):
            if (coin, tf) in KOMBIS:
                df = d[tf]
                atr = atr_von(df)
                markt[(coin, tf)] = (df, atr, doppel_signale(df["High"].values, df["Low"].values,
                                                             df["Close"].values, atr))

    meldungen = []

    # 1. Offene Trades auswerten
    log = auswerten(log, markt, meldungen)

    # 2. Neue Signale der letzten 24 Stunden
    kandidaten = []
    for (coin, tf), (df, atr, sig) in markt.items():
        for i in np.where(sig != 0)[0]:
            schluss = df.index[i] + REGELN[tf]["dauer"]
            if schluss < jetzt - pd.Timedelta(hours=24):
                continue
            if df.index[i].dayofweek >= 5 or schluss.dayofweek >= 5:
                continue  # keine neuen Trades am Wochenende
            kandidaten.append((schluss, coin, tf, i))

    for schluss, coin, tf, i in sorted(kandidaten):
        df, atr, sig = markt[(coin, tf)]
        sid = f"{coin}_{tf}_{df.index[i].isoformat()}"
        if sid in set(log["ID"]):
            continue
        offen = log[log["Status"] == "offen"]
        if ((offen["Coin"] == coin) & (offen["Chart"] == tf)).any():
            continue
        if len(offen) >= MAX_OFFEN:
            gemeldet = status.setdefault("uebersprungen", [])
            if sid not in gemeldet:  # jedes übersprungene Signal nur einmal melden
                meldungen.append(f"⏸ {coin} {tf}: Signal übersprungen, schon {MAX_OFFEN} Trades offen")
                gemeldet.append(sid)
                status["uebersprungen"] = gemeldet[-200:]
            continue
        s, ein = int(sig[i]), float(df["Close"].iloc[i])
        sl_d = SL_ATR * atr[i] / ein
        sl, tp = ein * (1 - s * sl_d), ein * (1 + s * 3 * sl_d)
        risiko_d = kontostand(log) * RISIKO
        position = risiko_d / sl_d
        hebel = int(max(1, min(REGELN[tf]["max_hebel"], np.floor(1 / (1.5 * sl_d + MM)))))
        liq = ein * (1 - s * (1 / hebel - MM))
        zeile = {"ID": sid, "Coin": coin, "Chart": tf, "Richtung": "LONG" if s == 1 else "SHORT",
                 "Signal_Kerze": df.index[i].isoformat(), "Einstieg_Zeit": schluss.isoformat(),
                 "Einstieg": ein, "SL": sl, "TP": tp, "Stop_pct": sl_d * 100,
                 "Risiko_$": round(risiko_d, 2), "Position_$": round(position, 2), "Hebel": hebel,
                 "Margin_$": round(position / hebel, 2), "Liq_Preis": liq, "Status": "offen",
                 "Ausstieg_Zeit": None, "Ausstieg": None, "R": None, "Ergebnis_$": None}
        log = pd.concat([log, pd.DataFrame([zeile])], ignore_index=True)
        tag = " (Beobachtung)" if (coin, tf) in BEOBACHTUNG else ""
        pfeil = "🟢" if s == 1 else "🔴"
        meldungen.append(f"{pfeil} NEUER DEMO-TRADE: {coin} {tf}{tag} {zeile['Richtung']}\n"
                         f"Einstieg: {ein:,.4f}\n"
                         f"Stop-Loss: {sl:,.4f} ({sl_d * 100:.2f} %)\n"
                         f"Take-Profit: {tp:,.4f}\n"
                         f"Position: {position:,.2f} $ | Hebel {hebel}x | Margin {position / hebel:,.2f} $\n"
                         f"Liquidation ca.: {liq:,.4f} | Risiko: {risiko_d:,.2f} $")

    # Neue, ältere Signale gleich nachträglich auswerten
    log = auswerten(log, markt, meldungen)
    log.to_csv(LOGDATEI, index=False)

    for m in meldungen:
        telegram(m)

    # Wochenbilanz: Freitag ab 22 Uhr oder Samstag, einmal pro Woche
    iso = jetzt.isocalendar()
    woche = f"{iso[0]}-W{iso[1]:02d}"
    if (jetzt.dayofweek == 4 and jetzt.hour >= 22) or jetzt.dayofweek == 5:
        if status.get("letzte_wochenbilanz") != woche:
            telegram(zusammenfassung(log, markt, f"📅 Wochenbilanz {woche}"))
            status["letzte_wochenbilanz"] = woche

    if TEST:
        telegram(zusammenfassung(log, markt, "✅ Paper-Bot läuft! Test-Nachricht"))

    status["letzter_lauf"] = jetzt.isoformat()
    speichere_status(status)
    print("Fertig.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status = lade_status()
        heute = pd.Timestamp.now(tz="Europe/Berlin").strftime("%Y-%m-%d")
        if status.get("letzter_fehler_tag") != heute:  # höchstens eine Fehlermeldung pro Tag
            telegram(f"⚠️ Paper-Bot Fehler: {type(e).__name__}: {e}")
            status["letzter_fehler_tag"] = heute
            speichere_status(status)
        raise
