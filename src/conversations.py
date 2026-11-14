"""
Agent-to-agent conversations (MiroFish-style "internal chat") + reasoning analysis for the report.

    python conversation_sim.py --project-dir . [--product ASIN] [--price-pct 0] [--marketing 0] [--comp-pct 0]
                               [--agents 400] [--seed 7] [--append-report]

Needs the outputs of phase3_build.py (and gradio_app.py in the same folder, which holds the data loader).

How a conversation works (every number is logged, so each opinion change can be traced and audited)
  * Each round agents are paired at random. In an exchange the SPEAKER states its concern (its top aspect) and cites a real
    retrieved review about it; the LISTENER reacts and updates its buy intention (in logit space):
        weight w = min(0.9, susceptibility x speaker_credibility x relevance_of_that_aspect_to_listener)
        new = old + w * (speaker_opinion - old) + evidence_weight * w * evidence_signal      (evidence_signal = (stars - 3) / 2)
  * Credibility, relevance mapping and the evidence weight are ASSUMED model parameters (listed in the report).
  * Sentences are templates built from the logged numbers. Optionally an LLM can voice the listener's reply
    (LLM_API_KEY / OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL); it only receives the logged facts.

Methods and why
  * Random pairing each round: the simplest interaction network, with no hidden structure that could drive the result. (Homophily or
    influencer-centred networks are natural extensions and are listed in the roadmap.)
  * The speaker cites a REAL retrieved review: every statement is grounded in the dataset, so a conversation can never invent a product fact.
  * Update rule with explicit terms (susceptibility x credibility x relevance, plus an evidence term): each opinion change is explainable
    and logged, so the reasoning can be audited. The mapping of influence to credibility and the evidence weight are ASSUMED.
  * Logit space for opinions: keeps intentions between 0 and 1 and makes equal logit steps comparable at high and low probabilities.
  * Integrity checks (citation rate, review-ID existence, reproducing every update from its logged terms): they verify TRACEABILITY.
    They do not validate the model against real consumers.
  * Template sentences by default (deterministic and free); an optional LLM can phrase the replies, receiving only the logged facts.
"""
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import make_agents, SCENARIOS, ASSUMED, A
from loaders import load_state, calibrate_b0

PARAMS = dict(evidence_weight=0.5, max_weight=0.9, rounds=ASSUMED["peer_rounds"],
              credibility="0.4 + 0.6*(influence-1)/2  (normal 0.40, reviewer 0.55, influencer 1.00)")
sig = lambda x: 1 / (1 + np.exp(-x))
cred = lambda infl: 0.4 + 0.6 * (infl - 1) / 2
pretty = lambda s: str(s).replace("_", " ")
LABEL = dict(budget_conscious="budget-conscious buyer", technology_enthusiast="technology enthusiast", professional_user="professional user", brand_loyal="brand-loyal customer",
             price_sensitive="price-sensitive buyer", camera_focused="camera-focused buyer", gamer="gamer", influencer="influencer",
             product_reviewer="product reviewer", competitor_customer="competitor's customer")
label = lambda s: LABEL.get(str(s), pretty(s))        # readable noun phrase for sentences ("a price-sensitive buyer")
art = lambda s: "an" if str(s)[:1].lower() in "aeiou" else "a"      # "an influencer", "a gamer"

def snippet(text, n=24):
    w = re.sub(r"\s+", " ", str(text)).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")

def llm_voice(facts):
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key: return None
    try:
        from openai import OpenAI
        c = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)
        r = c.chat.completions.create(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.7, max_tokens=110, messages=[
            {"role": "system", "content": "You voice a simulated consumer in a market simulation. Reply in 1-2 first-person sentences using ONLY the facts given. Do not invent product facts, prices or numbers."},
            {"role": "user", "content": facts}])
        return r.choices[0].message.content.strip()
    except Exception:
        return None

def run_conversations(S, pid, scen, n_agents=400, k=0.0, seed=7, use_llm=False, llm_cap=40):
    rng = np.random.default_rng(seed)
    ag = make_agents(S["personas"], int(n_agents), np.random.default_rng(seed))
    b0 = calibrate_b0(S, ag, k)
    dev = (ag["W"] * (S["SAT"][pid] - S["G"])).sum(1)
    o = b0 + ag["off"] + k * dev                                        # same starting opinions as the crowd model
    o = o - ASSUMED["price_elasticity"] * np.log(scen["price"]) * ag["psens"]
    o = o + ASSUMED["marketing_boost"] * scen["marketing"] * (1 - ag["s"])
    o = o - ASSUMED["competitor_switch"] * np.log(1 / scen["comp"]) * ag["aff"]
    names = np.array(ag["names"])[ag["pi"]]; n = len(o)
    ev = {}
    for i, asp in enumerate(A):
        h = S["ret"].search(asp.replace("_", " "), k=20, product_id=pid); ev[i] = h[h.score > 0].reset_index(drop=True)
    log, traj, llm_calls = [], [sig(o).copy()], 0
    for r in range(1, PARAMS["rounds"] + 1):
        perm = rng.permutation(n)
        for a_, b_ in zip(perm[::2], perm[1::2]):
            for sp, ls, turn in ((a_, b_, 1), (b_, a_, 2)):
                ai = int(ag["top"][sp]); h = ev[ai]
                if len(h):
                    row = h.iloc[int(rng.integers(len(h)))]; stars, rid, txt = int(row.rating), row.review_id, snippet(row.text)
                else:
                    stars, rid, txt = None, None, ""
                e = 0.0 if stars is None else (stars - 3) / 2
                rel = float(np.clip(ag["W"][ls, ai], 0.3, 2.0)); c = cred(ag["infl"][sp]); s = float(ag["s"][ls])
                w = min(PARAMS["max_weight"], s * c * rel); before, spk = o[ls], o[sp]
                social, evid = w * (spk - before), PARAMS["evidence_weight"] * w * e
                o[ls] = before + social + evid; after = o[ls]
                gap, moved = spk - before, after - before
                reaction = "agree" if abs(gap) < 0.15 else ("persuaded" if moved * gap > 0 and abs(moved) >= 0.02 else "unmoved")
                says = (f'As {art(label(names[sp]))} {label(names[sp])}, my main concern is {pretty(A[ai])}. A reviewer wrote "{txt}" ({stars} stars), which fits my {sig(spk):.0%} buy intention.'
                        if stars is not None else f"As {art(label(names[sp]))} {label(names[sp])}, my main concern is {pretty(A[ai])}; I found no review on it, so I rely on my own view ({sig(spk):.0%}).")
                imp = "more" if rel > 1.1 else "less" if rel < 0.9 else "about as much"
                cl = "very credible" if c >= .9 else "fairly credible" if c >= .5 else "not especially credible"
                if reaction == "agree": reply = f"I already see it much the same way ({sig(before):.0%}); little changes."
                elif reaction == "persuaded": reply = f"{pretty(A[ai]).capitalize()} matters {imp} to me (relevance {rel:.2f}) and {art(label(names[sp]))} {label(names[sp])} seems {cl} (credibility {c:.2f}), so I move from {sig(before):.1%} to {sig(after):.1%}."
                else: reply = f"I hear it, but I'm not persuaded: I'm not very open to it (susceptibility {s:.2f}, relevance {rel:.2f}), so I stay near {sig(after):.1%}."
                if use_llm and llm_calls < llm_cap:
                    v = llm_voice(f"You are {art(label(names[ls]))} {label(names[ls])}. {art(label(names[sp])).capitalize()} {label(names[sp])} said: {says} Your buy intention {sig(before):.1%} -> {sig(after):.1%}; reaction: {reaction}; relevance {rel:.2f}; credibility {c:.2f}.")
                    llm_calls += 1
                    if v: reply = v
                log.append(dict(round=r, turn=turn, speaker=int(sp), speaker_persona=names[sp], listener=int(ls), listener_persona=names[ls], aspect=A[ai],
                                evidence_review_id=rid, evidence_stars=stars, evidence_text=txt, speaker_says=says, listener_replies=reply, reaction=reaction,
                                speaker_logit=spk, logit_before=before, logit_after=after, listener_before=sig(before), listener_after=sig(after),
                                relevance=rel, credibility=c, susceptibility=s, weight=w, evidence_signal=e, social_term=social, evidence_term=evid))
        traj.append(sig(o).copy())
    return dict(log=pd.DataFrame(log), traj=np.array(traj), names=names, b0=b0, product=pid, scen=scen, n_agents=n, llm_used=llm_calls > 0)

def analyse(res, valid_ids=None):
    L = res["log"].copy(); an = {}
    L["shift_toward"] = (L.logit_after - L.logit_before) * np.sign(L.speaker_logit - L.logit_before)
    recomputed = L.logit_before + L.weight * (L.speaker_logit - L.logit_before) + PARAMS["evidence_weight"] * L.weight * L.evidence_signal
    an["integrity"] = pd.DataFrame([
        dict(check="messages that cite a retrieved real review", result=f"{L.evidence_review_id.notna().mean():.1%}"),
        dict(check="cited review IDs that exist in the review index", result=("n/a" if valid_ids is None else f"{L.evidence_review_id.dropna().isin(valid_ids).mean():.1%}")),
        dict(check="opinion updates reproducible from the logged terms (max abs error)", result=f"{float((recomputed - L.logit_after).abs().max()):.2e}"),
        dict(check="total logged messages", result=f"{len(L):,}")])
    an["influence"] = L.groupby("speaker_persona").agg(messages=("round", "size"), mean_shift_toward_speaker=("shift_toward", "mean"),
                        persuaded_pct=("reaction", lambda s: 100 * (s == "persuaded").mean())).round(3).sort_values("mean_shift_toward_speaker", ascending=False)
    L["abs_change_pts"] = (L.listener_after - L.listener_before).abs() * 100
    an["receptiveness"] = L.groupby("listener_persona").agg(exchanges=("round", "size"), mean_abs_change_pts=("abs_change_pts", "mean"),
                        persuaded_pct=("reaction", lambda s: 100 * (s == "persuaded").mean()),
                        agree_pct=("reaction", lambda s: 100 * (s == "agree").mean())).round(3).sort_values("mean_abs_change_pts", ascending=False)
    an["aspects"] = L.groupby("aspect").agg(messages=("round", "size"), mean_cited_stars=("evidence_stars", "mean"), mean_shift_toward_speaker=("shift_toward", "mean")).round(3).sort_values("messages", ascending=False)
    L["evidence_type"] = pd.cut(L.evidence_stars, [0, 2, 3, 5], labels=["negative (1-2 stars)", "mixed (3 stars)", "positive (4-5 stars)"])
    an["evidence_effect"] = L.dropna(subset=["evidence_type"]).groupby("evidence_type", observed=True).agg(messages=("round", "size"), mean_evidence_term_logit=("evidence_term", "mean")).round(4)
    t = res["traj"]; an["convergence"] = pd.DataFrame(dict(round=range(len(t)), mean_intention=t.mean(1).round(4), std_intention=t.std(1).round(4)))
    an["reaction_mix"] = (L.reaction.value_counts(normalize=True) * 100).round(1)
    an["excerpts"] = L.reindex((L.logit_after - L.logit_before).abs().sort_values(ascending=False).index).head(5)
    an["log"] = L
    return an

def dialogue_markdown(L, rnd=None, persona=None, limit=30):
    d = L
    if rnd not in (None, "all"): d = d[d["round"] == int(rnd)]
    if persona not in (None, "all"): d = d[(d.speaker_persona == persona) | (d.listener_persona == persona)]
    out = []
    for r in d.head(int(limit)).itertuples():
        out.append(f"**Round {r.round}** | **{pretty(r.speaker_persona).title()}** (agent {r.speaker}) to **{pretty(r.listener_persona).title()}** (agent {r.listener}) | topic: *{pretty(r.aspect)}*\n\n"
                   f"> **Says:** {r.speaker_says}\n>\n> **Replies:** {r.listener_replies}\n>\n"
                   f"> *Reasoning terms: relevance {r.relevance:.2f}, credibility {r.credibility:.2f}, susceptibility {r.susceptibility:.2f}, weight {r.weight:.2f}; "
                   f"reaction: {r.reaction}; buy intention {r.listener_before:.1%} to {r.listener_after:.1%}*\n")
    return "\n".join(out) or "No exchanges match this filter."

def analysis_figure(an, res):
    L = an["log"]; f, ax = plt.subplots(2, 2, figsize=(12, 8)); t = res["traj"]; x = np.arange(len(t))
    ax[0, 0].plot(x, t.mean(1), marker="o", color="#4c78a8"); ax[0, 0].fill_between(x, t.mean(1) - t.std(1), t.mean(1) + t.std(1), alpha=.25, color="#4c78a8")
    ax[0, 0].set_title("Mean buy intention +/- 1 SD across agents"); ax[0, 0].set_xlabel("round")
    inf = an["influence"].sort_values("mean_shift_toward_speaker"); ax[0, 1].barh(inf.index.map(pretty), inf.mean_shift_toward_speaker, color="#72b7b2"); ax[0, 1].set_title("Influence: mean opinion shift toward the speaker (logit)")
    mix = pd.crosstab(L.listener_persona, L.reaction, normalize="index") * 100; mix = mix.reindex(columns=[c for c in ["agree", "persuaded", "unmoved"] if c in mix.columns])
    mix.index = mix.index.map(pretty); mix.plot.barh(stacked=True, ax=ax[1, 0], color=["#bbbbbb", "#4c78a8", "#e45756"][:mix.shape[1]]); ax[1, 0].set_title("How listeners reacted (%)"); ax[1, 0].set_xlabel("")
    ee = an["evidence_effect"]; ax[1, 1].bar(ee.index.astype(str), ee.mean_evidence_term_logit, color=["#e45756", "#bbbbbb", "#4c78a8"][:len(ee)]); ax[1, 1].axhline(0, color="k", lw=.8)
    ax[1, 1].set_title("Effect of the cited review on the listener (logit)"); ax[1, 1].tick_params(axis="x", rotation=15)
    plt.tight_layout(); return f

def drift_note(an):
    L = an["log"].dropna(subset=["evidence_stars"]); pos = (L.evidence_stars >= 4).mean(); soc = L.social_term.abs().mean(); evd = L.evidence_term.mean()
    return (f"Reading the trend: {pos:.0%} of the reviews agents cited are 4-5 stars, so the evidence term pushes opinions up on average ({evd:+.4f} logit per exchange, "
            f"against an average absolute social term of {soc:.4f}). Part of any upward drift in mean intention is therefore the positive skew of the review data, "
            "not evidence that conversations make a product more appealing. Spread across agents shrinks because paired agents move toward each other.")

def report_section(res, an):
    sc = res["scen"]; t = res["traj"]; fmt = lambda df: df.to_string()
    ex = []
    for r in an["excerpts"].itertuples():
        ex.append(f"- Round {r.round}, {pretty(r.speaker_persona)} to {pretty(r.listener_persona)} (topic {pretty(r.aspect)}): \"{r.speaker_says}\" Reply: \"{r.listener_replies}\"")
    return "\n".join([
        "## Agent conversations and reasoning analysis", "",
        f"- Product {res['product']} | agents: {res['n_agents']} | rounds: {PARAMS['rounds']} | scenario: price x{sc['price']:.2f}, marketing {sc['marketing']:.1f}, competitor price x{sc['comp']:.2f}",
        f"- Mean buy intention: {t[0].mean():.1%} before the conversations, {t[-1].mean():.1%} after; spread across agents (SD) {t[0].std():.3f} to {t[-1].std():.3f}",
        f"- Voice: {'LLM-phrased replies (first 40 calls), template otherwise' if res['llm_used'] else 'template sentences generated from the logged numbers (no LLM)'}", "",
        "### How the reasoning works", "",
        "Agents are paired at random each round. The speaker states its main concern and cites a retrieved real review; the listener updates its buy intention with the rule",
        "`new = old + w*(speaker - old) + evidence_weight*w*evidence_signal`, where `w = min(0.9, susceptibility x speaker credibility x relevance of the topic to the listener)`.",
        f"Assumed parameters: evidence_weight = {PARAMS['evidence_weight']}; credibility = {PARAMS['credibility']}; maximum w = {PARAMS['max_weight']}.", "",
        "### Integrity checks (can every statement be traced?)", "", fmt(an["integrity"]), "",
        "### Who influences whom", "", "Speakers:", "", fmt(an["influence"]), "", "Listeners:", "", fmt(an["receptiveness"]), "",
        "### Which topics drive the conversations", "", fmt(an["aspects"]), "",
        "### Effect of the evidence cited", "", fmt(an["evidence_effect"]), "",
        "### Reaction mix (% of exchanges)", "", fmt(an["reaction_mix"]), "",
        "### Convergence", "", fmt(an["convergence"]), "",
        drift_note(an), "",
        "### Most influential exchanges (largest opinion change)", "", *ex, "",
        "### Limits of this evidence", "",
        "- The update rule, credibility mapping and evidence weight are assumptions, not fitted to data; this part of the evaluation explains the model's behaviour and checks traceability. It does not validate predictions against real consumers.",
        "- Sentences are templates unless the LLM option was used, and agents cite only reviews that already exist in the dataset.",
        "- Conversation outcomes depend on the random pairing; the seed and parameters above reproduce them."])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="."); ap.add_argument("--product"); ap.add_argument("--agents", type=int, default=400); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--price-pct", type=float, default=0); ap.add_argument("--marketing", type=float, default=0); ap.add_argument("--comp-pct", type=float, default=0)
    ap.add_argument("--k", type=float, default=0.0); ap.add_argument("--llm", action="store_true"); ap.add_argument("--append-report", action="store_true")
    a = ap.parse_args(); out = os.path.join(a.project_dir, "outputs"); os.makedirs(out, exist_ok=True)
    S = load_state(a.project_dir); pid = a.product or S["ref"].sort_values("n_reviews", ascending=False).product_id.iat[0]
    scen = dict(price=1 + a.price_pct / 100, marketing=a.marketing, comp=1 + a.comp_pct / 100)
    res = run_conversations(S, pid, scen, a.agents, a.k, a.seed, a.llm); an = analyse(res, set(S["ret"].meta.review_id))
    an["log"].drop(columns=["evidence_type"], errors="ignore").to_csv(os.path.join(out, "conversation_log.csv"), index=False)
    analysis_figure(an, res).savefig(os.path.join(out, "conversation_analysis.png"), dpi=130)
    sec = report_section(res, an); open(os.path.join(out, "conversation_report.md"), "w", encoding="utf-8").write(sec)
    if a.append_report:
        rp = os.path.join(out, "report.md"); base = open(rp, encoding="utf-8").read() if os.path.exists(rp) else "# Simulation report\n"
        base = re.sub(r"\n?<!-- CONV-START -->.*<!-- CONV-END -->\n?", "\n", base, flags=re.S)
        open(rp, "w", encoding="utf-8").write(base.rstrip() + "\n\n<!-- CONV-START -->\n" + sec + "\n<!-- CONV-END -->\n")
    print(an["integrity"].to_string(index=False)); print("\n", an["influence"].to_string()); print("\n", an["convergence"].to_string(index=False))
    print("\n--- sample exchanges ---\n" + dialogue_markdown(an["log"], 1, None, 3)); print(f"\nSaved conversation_log.csv, conversation_report.md, conversation_analysis.png to {out}")

if __name__ == "__main__":
    main()
