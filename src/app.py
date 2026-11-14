"""
Gradio dashboard: scenario simulator + agent-to-agent conversations + reasoning analysis + evaluation + report.

    python src/app.py --project-dir .         # then open http://127.0.0.1:7860

Run knowledge.py first (and simulate.py / evaluate.py if you want the Evaluation and Report tabs filled).

Methods and why
  * Gradio: named in the project proposal; gives sliders, plots and tabs with no front-end code, so the audience can change a
    price or marketing value live and watch the result.
  * The Simulate tab uses the fast crowd-average model (same as simulate.py). The Agent conversations tab runs the pairwise
    dialogue model (conversations.py), where agents talk one to one and every opinion change is logged with its reasoning terms.
    They start from identical opinions, so they can be compared; they differ only in how opinions spread.
  * Every tab states that price, marketing and competitor effects are ASSUMED (the review data has no prices or sales), and the
    Evaluation tab shows the leakage-free result next to the demo so the dashboard cannot be mistaken for a validated predictor.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate import make_agents, simulate, SCENARIOS, ASSUMED
from loaders import load_state, calibrate_b0
from conversations import run_conversations, analyse, dialogue_markdown, analysis_figure

def run(S, product, price_pct, marketing, comp_pct, n_agents, k, seed, use_llm):
    pid = product.split(" | ")[0]
    ag = make_agents(S["personas"], int(n_agents), np.random.default_rng(int(seed)))
    b0 = calibrate_b0(S, ag, k)
    scen = dict(price=1 + price_pct / 100, marketing=float(marketing), comp=1 + comp_pct / 100)
    args = (S["SAT"][pid], S["G"], S["NUDGE"][pid], k, b0)
    p_base = simulate(ag, *args, SCENARIOS["baseline"]); p_scen = simulate(ag, *args, scen)
    seg = pd.DataFrame({"persona": np.array(ag["names"])[ag["pi"]], "baseline": p_base, "scenario": p_scen}).groupby("persona").mean().sort_values("scenario")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); y = np.arange(len(seg))
    ax[0].barh(y - .2, seg.baseline, .4, label="baseline", color="#bbbbbb"); ax[0].barh(y + .2, seg.scenario, .4, label="your scenario", color="#4c78a8")
    ax[0].set_yticks(y); ax[0].set_yticklabels(seg.index, fontsize=8); ax[0].set_xlim(max(0, seg.values.min() - .05), min(1, seg.values.max() + .03)); ax[0].legend(); ax[0].set_title("Buy intention by persona")
    ax[1].hist(p_base, bins=25, alpha=.6, color="#bbbbbb", label="baseline"); ax[1].hist(p_scen, bins=25, alpha=.6, color="#4c78a8", label="your scenario")
    ax[1].set_title("Distribution of agents' buy intention"); ax[1].legend(); plt.tight_layout()
    d = (p_scen.mean() - p_base.mean()) * 100
    summary = (f"### {product}\n**Baseline buy intention:** {p_base.mean():.1%}  \n**Your scenario:** {p_scen.mean():.1%}  (**{d:+.2f} points**)  \n"
               f"Agents: {int(n_agents)} | product-signal strength k = {k} | fitted intercept = {b0:.2f}\n\n"
               f"*Price, marketing and competitor effects come from assumed parameters (price elasticity {ASSUMED['price_elasticity']}, "
               f"marketing boost {ASSUMED['marketing_boost']}, competitor switching {ASSUMED['competitor_switch']}), not from data: the reviews contain no prices or sales.*")
    rows = []
    for name, sc in SCENARIOS.items():
        pp = simulate(ag, *args, sc); rows.append(dict(scenario=name, buy_intention=f"{pp.mean():.1%}", change_pts=round((pp.mean() - p_base.mean()) * 100, 2)))
    # ---- pairwise conversations (the "internal chat") and their reasoning analysis
    res = run_conversations(S, pid, scen, int(n_agents), k, int(seed), bool(use_llm)); an = analyse(res, set(S["ret"].meta.review_id)); L = an["log"]; t = res["traj"]
    conv_md = (f"**Exchanges logged:** {len(L):,} over {int(L['round'].max())} rounds  \n**Mean buy intention:** {t[0].mean():.1%} before the conversations, {t[-1].mean():.1%} after  \n"
               f"**Spread across agents (SD):** {t[0].std():.3f} before, {t[-1].std():.3f} after  \n"
               f"**Reactions:** " + ", ".join(f"{k_} {v:.0f}%" for k_, v in an["reaction_mix"].items()) + "  \n"
               f"*Sentences are {'LLM-phrased (first calls)' if res['llm_used'] else 'templates built from the logged numbers'}. The update rule and its weights are assumptions.*")
    excerpts = dialogue_markdown(an["excerpts"], None, None, 5)
    return (summary, fig, pd.DataFrame(rows), conv_md, L, dialogue_markdown(L, 1, None, 25), analysis_figure(an, res),
            an["integrity"], an["influence"].reset_index(), an["receptiveness"].reset_index(), an["aspects"].reset_index(), an["evidence_effect"].reset_index(), excerpts)

def read_text(path, default):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else default

def build(project_dir):
    import gradio as gr
    S = load_state(project_dir)
    choices = [f"{r.product_id} | {int(r.n_reviews)} reviews | {r.positive_share:.1%} positive" for r in S["ref"].sort_values("n_reviews", ascending=False).itertuples()]
    personas = ["all"] + list(S["personas"])
    with gr.Blocks(title="Consumer response simulator") as demo:
        gr.Markdown(f"# Multi-agent consumer response simulator\nCategory: **{S['category']}** | personas calibrated from real Amazon reviews | "
                    "exploratory scenario tool, **not a validated sales predictor** (see the Evaluation tab).")
        log_state = gr.State()
        with gr.Tab("Simulate"):
            with gr.Row():
                with gr.Column(scale=1):
                    prod = gr.Dropdown(choices, value=choices[0], label="Product (ASIN)")
                    price = gr.Slider(-30, 30, 0, step=1, label="Our price change (%)")
                    mkt = gr.Slider(0, 1, 0, step=0.1, label="Marketing strength")
                    comp = gr.Slider(-30, 30, 0, step=1, label="Competitor price change (%)")
                    n = gr.Slider(100, 1000, 400, step=50, label="Number of agents")
                    k = gr.Slider(0, 5, 0, step=0.5, label="Product-signal strength k (0 = products treated alike)")
                    seed = gr.Number(7, label="Random seed", precision=0)
                    use_llm = gr.Checkbox(False, label="LLM phrases the replies (needs LLM_API_KEY; falls back to templates)")
                    btn = gr.Button("Run simulation", variant="primary")
                with gr.Column(scale=2):
                    out_md = gr.Markdown(); out_plot = gr.Plot(); out_tab = gr.Dataframe(label="Preset scenarios for this product")
        with gr.Tab("Agent conversations"):
            gr.Markdown("Agents are paired each round. The speaker states its main concern and cites a real review; the listener reacts and updates. "
                        "Press **Run simulation** first, then filter the exchanges below.")
            conv_md = gr.Markdown()
            with gr.Row():
                f_round = gr.Dropdown(["all", "1", "2", "3", "4", "5"], value="1", label="Round")
                f_persona = gr.Dropdown(personas, value="all", label="Persona (speaker or listener)")
                f_limit = gr.Slider(5, 100, 25, step=5, label="Exchanges shown")
            dialogue = gr.Markdown()
        with gr.Tab("Reasoning analysis"):
            gr.Markdown("The reasoning part of the evaluation: can every statement be traced to a real review and a logged rule, who influences whom, and what drives the conversations.")
            an_plot = gr.Plot(); t_int = gr.Dataframe(label="Integrity checks"); t_inf = gr.Dataframe(label="Influence by speaker persona")
            t_rec = gr.Dataframe(label="Receptiveness by listener persona"); t_asp = gr.Dataframe(label="Topics driving the conversations"); t_ev = gr.Dataframe(label="Effect of the evidence cited")
            gr.Markdown("### Most influential exchanges"); excerpts = gr.Markdown()
        with gr.Tab("Evaluation"):
            gr.Markdown(read_text(os.path.join(project_dir, "outputs", "negative_eval.md"), "Run `python src/evaluate.py --project-dir .` to show the evaluation here."))
        with gr.Tab("Report"):
            gr.Markdown(read_text(os.path.join(project_dir, "outputs", "report.md"), "Run `python src/simulate.py --project-dir .` to show the report here."))
        btn.click(lambda *a: run(S, *a), [prod, price, mkt, comp, n, k, seed, use_llm],
                  [out_md, out_plot, out_tab, conv_md, log_state, dialogue, an_plot, t_int, t_inf, t_rec, t_asp, t_ev, excerpts])
        refilter = lambda L, r, p, lim: "Run the simulation first." if L is None else dialogue_markdown(L, r, None if p == "all" else p, lim)
        for c in (f_round, f_persona, f_limit): c.change(refilter, [log_state, f_round, f_persona, f_limit], dialogue)
    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--project-dir", default="."); ap.add_argument("--share", action="store_true")
    a = ap.parse_args(); build(a.project_dir).launch(share=a.share)
