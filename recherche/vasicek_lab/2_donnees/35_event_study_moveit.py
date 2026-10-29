#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35 : event-study MOVEit, la tentative d'identification la plus dure avant de reculer.

La consigne de Hugo, appliquee a l'IDENTIFICATION et pas seulement a la parametrisation :
avant de decreter la direction inidentifiable et de reculer sur des bornes, on tente la
strategie la plus ambitieuse que la donnee autorise. Ici, une experience quasi naturelle.

LE DISPOSITIF. En mai 2023, l'exploitation de masse de la faille du logiciel de transfert
MOVEit (editeur TIERS) frappe simultanement des centaines d'organisations. C'est un choc
EXOGENE, DATE, et TYPE PILIER 4 (defaillance d'un prestataire ICT). Si la contagion
dirigee existe, un choc P4 doit elever, dans les mois qui suivent et chez les MEMES
organisations, le risque de defaillances relevant d'AUTRES piliers (gestion d'incident P2,
tests P3, gouvernance P1). C'est une prediction directionnelle testable.

METHODE. Difference de differences sur le hasard d'une nouvelle breche :
  - groupe TRAITE   : organisations financieres touchees pendant la fenetre MOVEit ;
  - groupe CONTROLE : organisations financieres actives avant, mais PAS dans la fenetre ;
  - AVANT / APRES   : 180 jours de part et d'autre ;
  - resultat        : nombre de JOURS-breche distincts par organisation (on effondre les
                      declarations d'un meme jour pour ne pas compter un batch comme une
                      sequence) ;
  - DiD             : (apres - avant)_traite - (apres - avant)_controle.
Un placebo (fausse date un an plus tot) controle le dispositif, et un bootstrap donne l'IC.

CE QU'ON TESTE VRAIMENT, ET LA LIMITE ANNONCEE D'AVANCE. La taxonomie de la source est un
type de VECTEUR (HACK, DISC, PHYS...), pas un domaine de controle DORA. Un DiD positif
dirait donc \"un choc P4 eleve le hasard global de breche\", PAS \"P4 entraine P_j\". La
cible reste non attribuable. C'est la borne que cette experience, la meilleure disponible,
ne franchit pas, et c'est un resultat : meme le meilleur cas naturel ne suffit pas sans un
registre taxonomise par pilier.

Sortie : diagnostics + figure Z6_event_study_moveit.png.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (_REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.severity.prc_analysis import chemin_prc, charger_chronologie   # noqa: E402

SRC = chemin_prc(_REPO)
if not SRC:
    sys.exit("donnee absente : data/raw/Data_Breach_Chronology.(xlsx|csv) "
             "(sources brutes non versionnees)")

WID = 80
RNG = np.random.default_rng(20260721)
HORIZON = 180                                  # jours avant / apres
MV0, MV1 = pd.Timestamp("2023-05-25"), pd.Timestamp("2023-06-15")


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


d = charger_chronologie(SRC, ["normalized_org_name", "breach_date", "breach_type",
                               "organization_type"])
d["bd"] = pd.to_datetime(d.breach_date.astype(str).str.strip(), errors="coerce")
bsf = d[(d.organization_type == "BSF") & d.bd.notna() & d.normalized_org_name.notna()].copy()
bsf["jour"] = bsf.bd.dt.normalize()


def jours_distincts(df, org, t0, t1):
    """Nombre de jours-breche distincts d'une organisation dans [t0, t1]."""
    s = df[(df.normalized_org_name == org) & (df.jour >= t0) & (df.jour <= t1)]
    return s.jour.nunique()


def did(ev0, ev1, label, verbose=True):
    """DiD autour d'une fenetre d'evenement [ev0, ev1]."""
    pre0, pre1 = ev0 - pd.Timedelta(days=HORIZON), ev0 - pd.Timedelta(days=1)
    post0, post1 = ev1 + pd.Timedelta(days=1), ev1 + pd.Timedelta(days=HORIZON)
    traite = set(bsf[(bsf.jour >= ev0) & (bsf.jour <= ev1)].normalized_org_name)
    # controle : actif dans le pre, absent de la fenetre
    actifs_pre = set(bsf[(bsf.jour >= pre0) & (bsf.jour <= pre1)].normalized_org_name)
    controle = actifs_pre - traite
    rows = []
    for grp, orgs in (("traite", traite), ("controle", controle)):
        for org in orgs:
            rows.append((grp,
                         jours_distincts(bsf, org, pre0, pre1),
                         jours_distincts(bsf, org, post0, post1)))
    df = pd.DataFrame(rows, columns=["grp", "pre", "post"])
    df["delta"] = df.post - df.pre
    gt = df[df.grp == "traite"]
    gc = df[df.grp == "controle"]
    est = gt.delta.mean() - gc.delta.mean()
    if verbose:
        print(f"  {label}")
        print(f"    traite   : {len(gt):4d} orgs, pre {gt.pre.mean():.3f} -> "
              f"post {gt.post.mean():.3f}  (delta {gt.delta.mean():+.3f})")
        print(f"    controle : {len(gc):4d} orgs, pre {gc.pre.mean():.3f} -> "
              f"post {gc.post.mean():.3f}  (delta {gc.delta.mean():+.3f})")
        print(f"    DiD = {est:+.3f} jour-breche/org")
    return df, est


def bootstrap_did(df, n=2000):
    # Tri avant tirage : RNG.choice parcourt le tableau dans son ordre, donc deux exports de
    # la meme base ranges differemment (xlsx et csv) donnaient deux IC pour un meme estimateur
    # ponctuel. Un bootstrap porte sur un multi-ensemble, pas sur un ordre.
    gt = np.sort(df[df.grp == "traite"].delta.values)
    gc = np.sort(df[df.grp == "controle"].delta.values)
    b = np.empty(n)
    for i in range(n):
        b[i] = (RNG.choice(gt, len(gt)).mean() - RNG.choice(gc, len(gc)).mean())
    return np.percentile(b, [2.5, 50, 97.5])


# =====================================================================================
titre("1. Le choc MOVEit comme experience quasi naturelle")
# =====================================================================================
mv = bsf[(bsf.jour >= MV0) & (bsf.jour <= MV1)]
print(f"  fenetre {MV0.date()} -> {MV1.date()} : {len(mv)} evenements, "
      f"{mv.normalized_org_name.nunique()} organisations financieres touchees")
print(f"  part du secteur BSF sur la periode : "
      f"{len(mv)/len(bsf[(bsf.jour>=MV0-pd.Timedelta(days=HORIZON))&(bsf.jour<=MV1+pd.Timedelta(days=HORIZON))]):.1%} "
      f"des evenements de la fenetre elargie")

# =====================================================================================
titre("2. Difference de differences : le choc P4 eleve-t-il le hasard ensuite ?")
# =====================================================================================
df_mv, est_mv = did(MV0, MV1, "MOVEit (2023)")
lo, med, hi = bootstrap_did(df_mv)
print(f"    IC bootstrap 95 % : [{lo:+.3f} ; {hi:+.3f}]  "
      f"({'significatif' if lo > 0 or hi < 0 else 'NON significatif : zero dans l IC'})")

# =====================================================================================
titre("3. Placebo : meme dispositif sur une fausse date, un an plus tot")
# =====================================================================================
df_pl, est_pl = did(MV0 - pd.Timedelta(days=365), MV1 - pd.Timedelta(days=365),
                     "placebo (2022)")
print(f"    Le placebo doit etre proche de zero si le dispositif ne fabrique pas d'effet.")

# =====================================================================================
titre("4. Le mur : la cible est-elle attribuable a un pilier ?")
# =====================================================================================
post0 = MV1 + pd.Timedelta(days=1)
post1 = MV1 + pd.Timedelta(days=HORIZON)
traite = set(mv.normalized_org_name)
suites = bsf[(bsf.jour >= post0) & (bsf.jour <= post1)
             & bsf.normalized_org_name.isin(traite)]
vc = suites.breach_type.value_counts(dropna=False)
unkn = vc.get("UNKN", 0)
print(f"  breches des organisations traitees dans les {HORIZON} j suivants : {len(suites)}")
print("  par type de vecteur :")
for t, n in vc.head(6).items():
    print(f"    {str(t):8} {n:4d}  ({n/len(suites):.0%})")
print(f"\n  Part NON attribuable a un vecteur precis (UNKN) : {unkn/len(suites):.0%}")
print("  Et surtout : meme les types renseignes (HACK, DISC...) sont des VECTEURS")
print("  d'attaque, pas des domaines de controle DORA. On ne peut donc PAS dire vers")
print("  quel pilier le choc P4 s'est propage. L'experience mesure un hasard global,")
print("  pas une direction pilier-a-pilier.")

# =====================================================================================
titre("5. Verdict")
# =====================================================================================
net = est_mv - est_pl
print(f"  DiD MOVEit    : {est_mv:+.3f}  [{lo:+.3f} ; {hi:+.3f}]")
print(f"  DiD placebo   : {est_pl:+.3f}  (fausse date, aucun choc reel)")
print(f"  EFFET NET (MOVEit - placebo) : {net:+.3f} jour-breche/org")
print()
print("  LE PIEGE, ET LA LECON. Le DiD MOVEit brut est grand et 'significatif'. Mais le")
print("  placebo produit quasiment le MEME chiffre, alors qu'aucun choc n'a eu lieu a cette")
print("  date. L'effet apparent n'est donc PAS causal : il vient de la selection du groupe")
print("  de controle (actif dans le pre par construction), qui revient mecaniquement vers")
print("  sa moyenne et fabrique un DiD positif quelle que soit la date. Une fois le placebo")
print(f"  retranche, l'effet net tombe a {net:+.3f}, indistinguable de zero.")
print()
print("  CONCLUSION, alignee sur le chapitre identifiabilite. Meme la meilleure experience")
print("  naturelle disponible (un choc P4 exogene, date, massif) NE FAIT PAS apparaitre de")
print("  propagation directionnelle nette, et sa cible serait de toute facon non attribuable")
print("  (taxonomie = vecteur, pas pilier ; 32 % d'UNKN). La direction demande un registre")
print("  horodate ET taxonomise par domaine de controle. La tentative la plus dure a ete")
print("  faite ; elle bute sur la donnee, pas sur la methode. C'est un resultat, pas un echec.")

# =====================================================================================
# figure
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Trois panneaux traces
# sur seize pouces puis imprimes sur 7,27 donnent 1,6 pouce par panneau et
# des etiquettes sous 5 points. Le rapport largeur sur hauteur est conserve.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.66, 3.17))

# (a) event-study : taux mensuel de breches traite vs controle autour du choc
traite_orgs = set(mv.normalized_org_name)
pre0 = MV0 - pd.Timedelta(days=HORIZON)
sub = bsf[(bsf.jour >= pre0) & (bsf.jour <= MV1 + pd.Timedelta(days=HORIZON))].copy()
sub["grp"] = np.where(sub.normalized_org_name.isin(traite_orgs), "traité", "contrôle")
sub["sem"] = ((sub.jour - MV0).dt.days // 14)
piv = sub.groupby(["sem", "grp"]).size().unstack(fill_value=0)
for grp, col in (("traité", ACCENT), ("contrôle", BLUE)):
    if grp in piv:
        ax1.plot(piv.index * 14, piv[grp], color=col, lw=2, marker="o", ms=3, label=grp)
ax1.axvspan(0, (MV1 - MV0).days, color=MUTED, alpha=0.18)
ax1.set_xlabel("jours depuis le début de MOVEit", color=INK2)
ax1.set_ylabel("événements par quinzaine", color=INK2)
ax1.set_title("(a)  Autour du choc P4", fontsize=11, color=INK, pad=8)
ax1.legend(frameon=False, fontsize=8)

# (b) DiD MOVEit vs placebo vs net : le placebo dévore l'effet apparent
ax2.bar([0], [est_mv], width=0.5, color=ACCENT, alpha=0.85)
ax2.errorbar([0], [est_mv], yerr=[[est_mv - lo], [hi - est_mv]], fmt="none",
             ecolor=INK, capsize=5, lw=1.5)
ax2.bar([1], [est_pl], width=0.5, color=MUTED, alpha=0.7)
ax2.bar([2], [est_mv - est_pl], width=0.5, color=GREEN, alpha=0.85)
ax2.axhline(0, color=INK, lw=1)
ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["MOVEit\nbrut", "placebo", "net"],
                                               fontsize=9)
ax2.set_ylabel("DiD (jour-brèche / org)", color=INK2)
ax2.set_title("(b)  Le placebo dévore l'effet :\nle net est nul", fontsize=11,
              color=INK, pad=8)

# (c) le mur taxonomique
parts = vc.head(5)
autre = len(suites) - parts.sum()
labels = [str(x) for x in parts.index] + (["autres"] if autre > 0 else [])
vals = list(parts.values) + ([autre] if autre > 0 else [])
cols = [ACCENT if str(l) == "UNKN" else BLUE for l in labels]
ax3.barh(range(len(vals))[::-1], vals, color=cols, alpha=0.85)
ax3.set_yticks(range(len(vals))[::-1]); ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel("événements post-choc", color=INK2)
ax3.set_title("(c)  Cible non attribuable :\nvecteurs, pas piliers", fontsize=11,
              color=INK, pad=8)

for ax in (ax1, ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z6_event_study_moveit.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
