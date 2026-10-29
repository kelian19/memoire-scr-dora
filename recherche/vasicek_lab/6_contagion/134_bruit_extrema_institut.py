#!/usr/bin/env python3
"""Bruit Monte-Carlo des extrema sur les 1 024 sommets, à paramètres fixés.

Chaque graine réévalue toute la grille avec des nombres communs entre matrices.
Les extrema sont sélectionnés à nouveau : le contrôle inclut les changements de
matrice extrémale. Les écarts-types entre graines ne sont pas des IC sur la
calibration et ne certifient pas les extrema du pavé continu.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB))
import partial_id as pid


def resumer(capitals):
    """Re-sélectionne les extrema pour chaque réplication de la grille."""
    q = np.asarray(capitals, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2 or not np.isfinite(q).all():
        raise ValueError("Au moins deux grilles finies de même longueur sont requises")
    minimum, maximum = q.min(axis=1), q.max(axis=1)
    largeur = maximum - minimum
    return {
        "minimum": minimum.tolist(), "maximum": maximum.tolist(),
        "largeur": largeur.tolist(),
        "indices_minimum": q.argmin(axis=1).tolist(),
        "indices_maximum": q.argmax(axis=1).tolist(),
        "moyennes": [float(x.mean()) for x in (minimum, maximum, largeur)],
        "ecarts_types": [float(x.std(ddof=1)) for x in (minimum, maximum, largeur)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annees", type=int, default=40_000)
    ap.add_argument("--graines", type=int, nargs="+",
                    default=[20260721, 20260722, 20260723, 20260724])
    ap.add_argument("--sortie", type=Path,
                    default=REPO / "resultats/institut_approfondissements/134.json")
    args = ap.parse_args()
    if args.annees < 1 or len(set(args.graines)) < 2 or len(set(args.graines)) != len(args.graines):
        ap.error("Années positives et au moins deux graines distinctes requises")
    start = time.monotonic()
    expert = pid.expert_matrix()
    sym, _ = pid.decompose(expert)
    matrices, indices, cards = [], [], []
    for index, a in enumerate(pid.all_vertices(sym, 1.0)):
        w = pid.build_W(sym, a)
        if pid.admissible(w):
            matrices.append(w)
            indices.append(index)
            cards.append(pid.card_dist_all(w)[0])
    print(f"134 : {len(cards)} sommets admissibles ; {args.annees} années par graine", flush=True)
    capitals, expert_capitals = [], []
    for seed in args.graines:
        ev = pid.Evaluator(n_years=args.annees, seed=seed)
        row = []
        for k, card in enumerate(cards):
            row.append(ev.from_card(card)[0])
            if (k + 1) % 256 == 0:
                print(f"  graine {seed}, {k+1}/{len(cards)}", flush=True)
        capitals.append(row)
        expert_capitals.append(ev.scr(expert))
        print(f"  extrema [{min(row):.1f} ; {max(row):.1f}], largeur {max(row)-min(row):.1f} M EUR", flush=True)
        del ev
    summary = resumer(capitals)
    out = dict(annees=args.annees, graines=args.graines, n_sommets=len(cards),
               indices_sommets=indices, capitaux=capitals, capitaux_expert=expert_capitals,
               resume=summary, duree_secondes=time.monotonic()-start)
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Moyennes minimum / maximum / largeur : " + " / ".join(f"{x:.1f}" for x in summary["moyennes"]))
    print("Ecarts-types entre graines : " + " / ".join(f"{x:.1f}" for x in summary["ecarts_types"]))
    print("Ces écarts-types mesurent le bruit de simulation, à paramètres fixés.")
    print("Ils ne sont ni des IC de calibration ni une garantie sur le pavé continu.")


if __name__ == "__main__":
    main()
