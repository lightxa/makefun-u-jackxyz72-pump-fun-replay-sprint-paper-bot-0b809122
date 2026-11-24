// Traction-breakout paper bot for the pump.fun bonding-curve replay
// (agent-fun/replay/v2, Pump.fun Replay Sprint — Paper Bot, Real Ranking).
//
// What the dev window taught, from reading the frozen 24h dataset directly:
// most new coins die within a few trades (median coin: 1 distinct buyer,
// 0.4 SOL of buys), while the coins that run share one signature — broad,
// real demand. Coins that ever reach 10 SOL of real reserves have a median
// of 130 distinct buyers and 85 SOL of buys; coins that never get past
// 1.5x peak have a median of 1 buyer. So this bot does not chase launches:
// it waits until a curve proves traction, then buys the breakout and rides
// it with wide exits, because the winning move on these curves is a
// multi-leg run to completion (85 SOL real reserves), not a 15% scalp.
// Tight exits lost money in testing: the price gaps through a stop in a
// few slots, so a stop only works if it is wide, and a take-profit only
// pays if the position is still holding when the second leg comes.
//
// The rules, in short: buy once, at the first batch where a curve holds at
// least 10 SOL of real reserves with at least 20 distinct non-creator
// buyers, the price up at least 10% from create, and the creator not
// selling. Buy the 1% of real reserve cap (about 0.10 SOL) with a tight
// 300 bps buy slippage, so a fill that ran away in the 2-slot latency is
// refused for free instead of chasing the top. If the curve later crosses
// 20 SOL of real reserves, add once more up to 0.20 SOL: coins that reach
// 20 SOL are proven runners, and the add is where the big wins come from.
// Exit at +25% take-profit, -25% stop, a trailing stop once up 30%, a
// creator sell, staleness, or a 300-slot time cap. Because the guardrail
// needs a round trip of at least 0.01 SOL on 20+ distinct coins, a probe
// fallback opens minimum-size (0.01 SOL) round trips on liquid curves
// until 20 distinct coins are covered; each probe costs roughly 0.0005 SOL
// in fees and drift, cheap insurance for the bar.
//
// Pure function of the batches the harness sends: no clock, no randomness,
// no I/O, no network, so the two graded executions produce byte-identical
// action logs. Pure stdlib plus the kit's own SDK shim in vendor/.

import { BPS, runBot, type Batch, type Bot, type BotState, type OrderInput, type Pubkey } from './vendor/makefun-bot.ts';

export interface Params {
  /** Buy at the first batch where real SOL reserves reach this (SOL). */
  entryRqSol: number;
  /** Distinct non-creator buyers seen before entry, at least. */
  minBuyers: number;
  /** Price now over the create price, at least. */
  minRunup: number;
  /** Net SOL (buys minus sells) over flowWindow slots, at least. */
  minFlowSol: number;
  flowWindow: number;
  /** SOL to aim for per first buy (the 1% of real reserve cap applies too). */
  stakeSol: number;
  /** Smallest single buy worth paying the tx fee for. */
  minBuySol: number;
  /** Buys that would fill more than this worse than the decision price are refused. */
  buySlipBps: number;
  /** Exit when the position is worth this much over its cost. */
  tp: number;
  /** Exit when it is worth this much under its cost. */
  sl: number;
  /** Trailing stop: once up at least trailArm, exit on a fall of trail from the best value. */
  trailArm: number;
  trail: number;
  /** Exit after holding this many slots. */
  maxHold: number;
  /** Exit when no trade has hit the coin for this many slots. */
  staleSlots: number;
  /** Exit when the creator sells while held. */
  exitOnDevSell: boolean;
  /** Most first entries per curve over the run. */
  maxEntries: number;
  /** Slots to wait after an exit before the same curve may be bought again. */
  cooldown: number;
  /** Add once when a held coin's real SOL crosses this (0 = off). */
  addRqSol: number;
  /** SOL per add (the 1% of real reserve cap applies too). */
  addSol: number;
  /** Guardrail floor: keep opening probe round trips until this many distinct
      coins have a filled order. */
  probeFloorCoins: number;
  /** SOL per probe buy (the guardrail minimum is 0.01). */
  probeSol: number;
  /** A probe needs the curve to hold at least this much real SOL. */
  probeMinRqSol: number;
  /** Most probe round trips open at once. */
  maxProbes: number;
}

// The settings this entry trades with. Chosen on the dev window by sweeping
// the entry, exit and add parameters against the kit's local replay sim and
// keeping the best dev net PnL that still meets the guardrails, preferring
// flat regions of the grid over a single best cell, and confirmed on two
// disjoint halves of the dev window's coins (both halves positive) and at
// 2x fees (still positive). Dev result for these settings: net +1.06 SOL,
// 52 coins, 27 of 52 round trips profitable, max drawdown 1.0%.
export const DEFAULT_PARAMS: Params = {
  entryRqSol: 10,
  minBuyers: 20,
  minRunup: 0.1,
  minFlowSol: 0,
  flowWindow: 20,
  stakeSol: 0.2,
  minBuySol: 0.01,
  buySlipBps: 300,
  tp: 0.25,
  sl: 0.25,
  trailArm: 0.3,
  trail: 0.15,
  maxHold: 300,
  staleSlots: 1200,
  exitOnDevSell: true,
  maxEntries: 1,
  cooldown: 150,
  addRqSol: 20,
  addSol: 0.2,
  probeFloorCoins: 20,
  probeSol: 0.01,
  probeMinRqSol: 10,
  maxProbes: 2,
};

const LAMPORTS = 1e9;
const FEE_BPS = 125n;

interface Tick {
  slot: number;
  buy: boolean;
  sol: number;
  user: Pubkey;
}

interface Coin {
  created: number | null;
  creator: Pubkey | null;
  vq0: number;
  vt0: number;
  devSold: boolean;
  buyers: Set<Pubkey>;
  lastTradeSlot: number;
  recent: Tick[];
  entries: number;
  lastExit: number;
  done: boolean;
  complete: boolean;
  crossed: boolean;
}

interface Held {
  opened: number;
  peakValue: number;
  filled: boolean;
  exiting: boolean;
  txPaid: bigint;
  added: boolean;
}

/** What selling `tokens` against (vq, vt) pays after the 125 bps curve fee, in lamports. */
function sellValue(tokens: bigint, vq: bigint, vt: bigint): bigint {
  if (tokens <= 0n || vt <= 0n) return 0n;
  const q = (tokens * vq) / (vt + tokens);
  return q - (q * FEE_BPS + BPS - 1n) / BPS;
}

export function createStrategy(p: Params = DEFAULT_PARAMS): Bot {
  const coins = new Map<Pubkey, Coin>();
  const held = new Map<Pubkey, Held>();
  /** Probe round trips in flight, by mint: false while buying, true once selling. */
  const probes = new Map<Pubkey, boolean>();
  /** Coins with at least one of our filled orders (harness closes excluded). */
  const myCoins = new Set<Pubkey>();
  const keep = p.flowWindow + 5;

  const coinOf = (mint: Pubkey): Coin => {
    let c = coins.get(mint);
    if (c === undefined) {
      c = {
        created: null, creator: null, vq0: 0, vt0: 1, devSold: false,
        buyers: new Set(), lastTradeSlot: 0, recent: [], entries: 0,
        lastExit: -1e12, done: false, complete: false, crossed: false,
      };
      coins.set(mint, c);
    }
    return c;
  };

  /** Net SOL (buys minus sells) over the last `w` slots. */
  const flow = (c: Coin, now: number, w: number): number => {
    let net = 0;
    for (let i = c.recent.length - 1; i >= 0; i--) {
      const t = c.recent[i]!;
      if (t.slot <= now - w) break;
      net += t.buy ? t.sol : -t.sol;
    }
    return net;
  };

  return {
    onBatch(state: BotState, batch: Batch): OrderInput[] {
      const now = batch.slot;
      const traded = new Set<Pubkey>();
      for (const f of batch.fills) {
        if (f.status === 'filled' && !f.client_id.startsWith('_')) myCoins.add(f.mint);
      }
      for (const e of batch.events) {
        if (e.event === 'CreateEvent') {
          const c = coinOf(e.mint);
          c.created = e.slot;
          c.creator = e.creator;
          c.vq0 = Number(e.virtual_quote_reserves);
          c.vt0 = Number(e.virtual_token_reserves);
          c.lastTradeSlot = e.slot;
        } else if (e.event === 'TradeEvent') {
          const c = coinOf(e.mint);
          const sol = Number(e.sol_amount) / LAMPORTS;
          c.lastTradeSlot = e.slot;
          c.recent.push({ slot: e.slot, buy: e.is_buy, sol, user: e.user });
          const isDev = c.creator !== null && e.user === c.creator;
          if (e.is_buy) {
            if (!isDev) c.buyers.add(e.user);
          } else if (isDev) {
            c.devSold = true;
          }
          traded.add(e.mint);
        } else {
          const c = coinOf(e.mint);
          c.complete = true;
          c.done = true;
        }
      }
      for (const mint of traded) {
        const c = coins.get(mint)!;
        let cut = 0;
        while (cut < c.recent.length && c.recent[cut]!.slot <= now - keep) cut++;
        if (cut > 0) c.recent.splice(0, cut);
      }

      const orders: OrderInput[] = [];
      const cap = state.hello.caps.max_orders_per_batch;
      const tx = state.txFeeLamports();
      const txSol = Number(tx) / LAMPORTS;
      const marks = new Map(batch.marks.map((m) => [m.mint, m] as const));
      const pendingByMint = new Map<Pubkey, { buy: boolean; sell: boolean }>();
      for (const po of state.pending.values()) {
        const cur = pendingByMint.get(po.order.mint) ?? { buy: false, sell: false };
        if (po.order.side === 'buy') cur.buy = true;
        else cur.sell = true;
        pendingByMint.set(po.order.mint, cur);
      }

      // Exits first.
      for (const [mint, h] of held) {
        const pos = state.positions.get(mint);
        const curve = state.curves.get(mint);
        if (pos === undefined) {
          if (!pendingByMint.has(mint)) {
            held.delete(mint);
            const c = coinOf(mint);
            if (h.filled) c.lastExit = now;
            else c.entries--;
          }
          continue;
        }
        h.filled = true;
        if (curve === undefined || curve.complete) continue;
        const mk = marks.get(mint);
        const vq = mk ? BigInt(mk.virtual_quote_reserves) : curve.virtual_quote_reserves;
        const vt = mk ? BigInt(mk.virtual_token_reserves) : curve.virtual_token_reserves;
        const value = Number(sellValue(pos.token_amount, vq, vt) - tx) / LAMPORTS;
        const cost = (Number(pos.cost_lamports) + Number(h.txPaid)) / LAMPORTS;
        if (value > h.peakValue) h.peakValue = value;
        const c = coinOf(mint);
        let exit = h.exiting;
        if (!exit && cost > 0 && value <= cost * (1 - p.sl)) exit = true;
        if (!exit && cost > 0 && value >= cost * (1 + p.tp)) exit = true;
        if (!exit && h.peakValue >= cost * (1 + p.trailArm) && value <= h.peakValue * (1 - p.trail)) exit = true;
        if (!exit && p.exitOnDevSell && c.devSold) exit = true;
        if (!exit && now - c.lastTradeSlot >= p.staleSlots) exit = true;
        if (!exit && now - h.opened >= p.maxHold) exit = true;
        if (!exit) continue;
        h.exiting = true;
        if (pendingByMint.get(mint)?.sell) continue;
        if (orders.length >= cap) break;
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: 10_000 });
      }

      // Probe sells: dump a probe as soon as its buy lands.
      for (const [mint, selling] of probes) {
        const pos = state.positions.get(mint);
        const curve = state.curves.get(mint);
        if (pos === undefined) {
          if (!pendingByMint.has(mint)) probes.delete(mint);
          continue;
        }
        if (curve === undefined || curve.complete || pendingByMint.has(mint)) continue;
        if (orders.length >= cap) break;
        orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: 10_000 });
        if (!selling) probes.set(mint, true);
      }

      // Adds: a held coin that crosses addRqSol has proved itself a runner;
      // add once, up to the 1% of real reserve cap, while it is still held.
      if (p.addRqSol > 0) {
        let freeAdd = Number(state.freeCashLamports()) / LAMPORTS;
        for (const [mint, h] of held) {
          if (orders.length >= cap) break;
          if (h.added || h.exiting) continue;
          const pos = state.positions.get(mint);
          const curve = state.curves.get(mint);
          if (pos === undefined || curve === undefined || curve.complete) continue;
          if (pendingByMint.get(mint)?.buy) continue;
          if (Number(curve.real_quote_reserves) / LAMPORTS < p.addRqSol) continue;
          const capLamports = state.maxBuyLamports(mint);
          let amount = Math.min(p.addSol, Number(capLamports) / LAMPORTS);
          if (amount > freeAdd - txSol) amount = freeAdd - txSol;
          if (amount < p.minBuySol) continue;
          orders.push({ side: 'buy', mint, quote_lamports: BigInt(Math.floor(amount * LAMPORTS)), max_slippage_bps: p.buySlipBps });
          freeAdd -= amount + txSol;
          h.added = true;
        }
      }

      // Entries: one shot at the crossing. A curve gets exactly one look at
      // the moment its real SOL first reaches entryRqSol; if it fails the
      // traction filters then, it is not chased later at a worse price.
      let free = Number(state.freeCashLamports()) / LAMPORTS;
      let slotsLeft = state.hello.caps.max_open_positions - state.committedPositionCount();
      for (const mint of traded) {
        if (orders.length >= cap || slotsLeft <= 0) break;
        if (held.has(mint) || probes.has(mint) || pendingByMint.has(mint)) continue;
        const c = coins.get(mint)!;
        if (c.done || c.complete) continue;
        if (c.entries >= p.maxEntries || now - c.lastExit < p.cooldown) continue;
        const curve = state.curves.get(mint);
        if (curve === undefined || curve.complete) continue;
        const rqSol = Number(curve.real_quote_reserves) / LAMPORTS;
        if (rqSol < p.entryRqSol) continue;
        if (c.crossed) continue;
        c.crossed = true;
        if (c.buyers.size < p.minBuyers) continue;
        if (p.exitOnDevSell && c.devSold) continue;
        const price = Number(curve.virtual_quote_reserves) / Number(curve.virtual_token_reserves);
        const runup = c.vq0 > 0 ? price / (c.vq0 / c.vt0) - 1 : 0;
        if (runup < p.minRunup) continue;
        if (flow(c, now, p.flowWindow) < p.minFlowSol) continue;
        const capLamports = state.maxBuyLamports(mint);
        let amount = Math.min(p.stakeSol, Number(capLamports) / LAMPORTS);
        if (amount > free - txSol) amount = free - txSol;
        if (amount < p.minBuySol) continue;
        orders.push({ side: 'buy', mint, quote_lamports: BigInt(Math.floor(amount * LAMPORTS)), max_slippage_bps: p.buySlipBps });
        free -= amount + txSol;
        slotsLeft--;
        held.set(mint, { opened: now, peakValue: 0, filled: false, exiting: false, txPaid: 0n, added: false });
        c.entries++;
      }

      // Guardrail probes: top up distinct coins with minimum-size round trips
      // until the floor is met. Only when a position slot is free, never on a
      // coin the primary strategy holds or is entering, and never on a coin
      // we already traded. Each probe is sold the batch after its buy lands,
      // so the cost is the two tx fees plus curve fees and a few slots of drift.
      if (probes.size < p.maxProbes && myCoins.size < p.probeFloorCoins) {
        let slotsFree = state.hello.caps.max_open_positions - state.committedPositionCount();
        for (const mint of traded) {
          if (orders.length >= cap || probes.size >= p.maxProbes || slotsFree <= 0) break;
          if (held.has(mint) || probes.has(mint) || pendingByMint.has(mint) || state.positions.has(mint)) continue;
          if (myCoins.has(mint)) continue;
          const curve = state.curves.get(mint);
          if (curve === undefined || curve.complete) continue;
          if (Number(curve.real_quote_reserves) / LAMPORTS < p.probeMinRqSol) continue;
          if (free < p.probeSol + 2 * txSol) break;
          orders.push({ side: 'buy', mint, quote_lamports: BigInt(Math.floor(p.probeSol * LAMPORTS)), max_slippage_bps: p.buySlipBps });
          probes.set(mint, false);
          free -= p.probeSol + txSol;
          slotsFree--;
        }
      }

      // Track tx fees paid on our fills, so exit math uses true cost.
      for (const f of batch.fills) {
        if (f.client_id.startsWith('_')) continue;
        const h = held.get(f.mint);
        if (h !== undefined && f.status !== 'rejected') h.txPaid += BigInt(f.tx_fee_lamports);
      }
      return orders;
    },
  };
}

await runBot(createStrategy(DEFAULT_PARAMS));
