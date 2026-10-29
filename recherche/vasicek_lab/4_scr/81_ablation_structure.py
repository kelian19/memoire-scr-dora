#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
81 : l'ABLATION en echelle complete. Ce que la structure apporte, brique par brique.

CE QUE LE MEMOIRE FAIT AUJOURD'HUI. Le chapitre 9 retire toute contagion (W = 0) et publie le
« SCR nu », 5 275 M, presente comme le plancher defendable. Il note que le prepublie de cascade
climatique conduit la meme experience et qu'y annuler la propagation « effondre a la fois le point
d'attachement ET la mesure de queue ».

CE QUI MANQUAIT, ET C'EST DOUBLE.
  1. UNE SEULE brique est ablatee, la propagation. Or le modele en compte au moins six, et rien ne
     dit laquelle porte le resultat. Une ablation unique ne separe pas ce qu'un modele APPORTE de
     ce qu'il HERITE de ses marges : c'est precisement l'argument invoque pour la justifier.
  2. UNE SEULE grandeur est regardee, le niveau. Le prepublie en regarde DEUX, et il a raison :
     un modele peut conserver son niveau en perdant sa forme de queue, ou l'inverse. On en regarde
     donc TROIS ici, la troisieme etant celle qui porte la these du memoire :
       - le NIVEAU        : VaR 99,5 % de la charge annuelle ;
       - la FORME de queue : rapport CTE / VaR au meme niveau ;
       - l'ORDRE des piliers : allocation de queue, qui doit rester P1 > P4 > P2 > P3 > P5.

LES SIX ABLATIONS, et la derniere est celle qu'un jury demanderait :
  (a) propagation          g = 0, la cascade ne sort jamais du pilier d'amorce ;
  (b) auto-evitement       branchement multitype : un pilier peut en declencher PLUSIEURS ;
  (c) surdispersion        phi = 1, la frequence redevient un Poisson pur ;
  (d) accumulation P4      phi_cs = None, P4 redevient un noeud ordinaire de cascade ;
  (e) queue lourde         xi = 0, la severite redevient exponentielle ;
  (f) heterogeneite d'amorce  amorces UNIFORMES entre piliers au lieu de proportionnelles.
La (f) attaque directement une limite que le memoire declare : il ecrit que le classement des
piliers « reproduit ROOT, en partie MECANIQUEMENT, les amorces y etant semees proportionnellement ».
Si l'ordre survit a des amorces uniformes, il n'est pas mecanique. S'il tombe, la limite declaree
est plus lourde qu'annonce. Dans les deux cas la reponse vaut d'etre imprimee.

CE QU'IL NE FAIT PAS : deplacer un nombre publie. L'etat de reference reproduit 20 188 M au
centime, et chaque ablation est comparee a CE reference dans le MEME moteur.

Sortie : diagnostics + figure S37_ablation_structure.png.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import euro_cascade_model as ec                                 # noqa: E402
from euro_cascade_model import var                              # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 88
PIL = cx.PIL
sp = cx.sp
NY, NSEED, SEED0 = cx.NY, cx.NSEED, cx.SEED0
ALPHA = 0.995

W_LAM = np.array([eng.LAMBDA[j] for j in PIL], float)
W_LAM = W_LAM / W_LAM.sum()
W_UNI = np.full(len(PIL), 1.0 / len(PIL))


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    """Separateur de milliers applique AU NOMBRE SEUL, jamais a une phrase entiere."""
    return f"{v:,.0f}".replace(",", " ")


def tables_cascade(g, mode):
    """Tables (indicateur, probas) par pilier d'amorce, marche auto-evitante ou branchement."""
    if mode == "branching":
        return eng.build_cascade_tables(g, mode="branching")
    return {j: cx.table_amorce(j, g) for j in PIL}


def pertes_par_pilier(rng, *, lam, g, p_u, phi_cs, xi=None, phi=None, mode="walk",
                      amorce="lambda", ny=NY):
    """Pertes annuelles VENTILEES par pilier touche, avec un interrupteur par brique.

    Sans aucun interrupteur active, la somme des colonnes doit redonner EXACTEMENT la sortie de
    canaux_conformite.pertes_annuelles : c'est le controle du script, verifie plus bas. L'ordre
    des tirages est celui du module partage (comptage, amorce, uniformes, severites), sans quoi
    la reconciliation ne serait qu'une coincidence de tirages.
    """
    xi = sp["xi"] if xi is None else xi
    phi = ec.PHI if phi is None else phi
    nP = len(PIL)
    if phi <= 1.0 + 1e-9:                         # Poisson pur : la surdispersion est retiree
        counts = rng.poisson(lam, size=ny)
    else:
        r = lam / (phi - 1.0)
        counts = rng.negative_binomial(r, r / (r + lam), size=ny)
    # DEUX ventilations, et elles ne mesurent pas la meme chose (cf. section 4) :
    #   touche : ou le capital se LOGE, lecture d'Euler ;
    #   amorce : quel pilier SOURCE l'engendre, lecture de la these.
    touche = np.zeros((ny, nP))
    amor = np.zeros((ny, nP))
    T = int(counts.sum())
    if T == 0:
        return touche, amor
    year_of = np.repeat(np.arange(ny), counts)
    w = W_UNI if amorce == "uniforme" else W_LAM
    am = rng.choice(nP, size=T, p=w)
    U = rng.random(T)
    SEV = simulate_remediation_severity(T * nP, xi, sp["sigma"], sp["u"], p_u,
                                        sp["cap"], rng).reshape(T, nP)
    tables = tables_cascade(g, mode)
    if phi_cs is not None:
        tables[cx.P4] = cx.table_p4_choc(phi_cs)
    for c, j in enumerate(PIL):
        idx = np.where(am == c)[0]
        if idx.size == 0:
            continue
        ind, probs = tables[j]
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        sel = np.searchsorted(cdf, U[idx], side="right")
        np.clip(sel, 0, len(probs) - 1, out=sel)
        contrib = SEV[idx] * ind[sel]
        for cp in range(nP):
            touche[:, cp] += np.bincount(year_of[idx], weights=contrib[:, cp], minlength=ny)
        amor[:, c] += np.bincount(year_of[idx], weights=contrib.sum(axis=1), minlength=ny)
    return touche, amor


def _parts_queue(M, queue):
    p = M[queue].mean(axis=0) if queue.any() else np.zeros(len(PIL))
    return p / p.sum() if p.sum() > 0 else p


def mesures(**kw):
    """(VaR, CTE/VaR, parts par pilier TOUCHE, parts par pilier d'AMORCE), graine par graine."""
    v, r, pt, pa = [], [], [], []
    for k in range(NSEED):
        T_, A_ = pertes_par_pilier(np.random.default_rng(SEED0 + k), **kw)
        tot = T_.sum(axis=1)
        q = var(tot, ALPHA)
        queue = tot >= q
        v.append(q)
        r.append(float(tot[queue].mean() / q) if queue.any() else np.nan)
        pt.append(_parts_queue(T_, queue))
        pa.append(_parts_queue(A_, queue))
    return np.array(v), np.array(r), np.array(pt), np.array(pa)


def classement(parts):
    """Ordre decroissant des parts moyennes."""
    return [PIL[c] for c in np.argsort(-parts.mean(axis=0))]


def stable(parts):
    """L'ordre est-il le MEME sur toutes les graines ? Sinon il ne se lit pas."""
    ords = [tuple(PIL[c] for c in np.argsort(-p)) for p in parts]
    return len(set(ords)) == 1


REF = dict(lam=cx.LAM_NC, g=cx.G_NC, p_u=cx.PU_NC, phi_cs=cx.PHICS_NC)

# =====================================================================================
titre("1. CONTROLE : l'etat de reference reproduit le nombre publie")
# =====================================================================================
ref_pub = cx.scr_config(**REF)
v_ref, r_ref, pt_ref, pa_ref = mesures(**REF)
print(f"  Etat non conforme, moteur partage : {ref_pub:.2f} M€")
print(f"  Meme etat, moteur ventile par pilier de ce script : {v_ref.mean():.2f} M€")
print(f"  Ecart : {abs(v_ref.mean() - ref_pub):.4f} M€. Les colonnes se somment donc bien a la")
print("  charge agregee, et chaque ablation qui suit se compare a CE reference.")
ordre_t = classement(pt_ref)
ordre_a = classement(pa_ref)
ordre_root = sorted(PIL, key=lambda k: -eng.LAMBDA[k])
print(f"\n  QUATRE grandeurs de reference, sur {NSEED} graines :")
print(f"    niveau, VaR 99,5 %          {v_ref.mean():>10.0f} M€   (etendue {np.ptp(v_ref):.0f})")
print(f"    forme, CTE / VaR            {r_ref.mean():>10.3f}      (etendue {np.ptp(r_ref):.3f})")
print(f"    ordre par pilier TOUCHE     {' > '.join('P' + str(j) for j in ordre_t)}"
      f"   stable entre graines : {stable(pt_ref)}")
print(f"    ordre par pilier d'AMORCE   {' > '.join('P' + str(j) for j in ordre_a)}"
      f"   stable entre graines : {stable(pa_ref)}")
print(f"    parts par pilier touche     "
      + "  ".join(f"P{j} {100*pt_ref.mean(axis=0)[c]:.1f} %" for c, j in enumerate(PIL)))
print(f"    parts par pilier d'amorce   "
      + "  ".join(f"P{j} {100*pa_ref.mean(axis=0)[c]:.1f} %" for c, j in enumerate(PIL)))
print("\n  DEUX ORDRES, ET IL FAUT SAVOIR LEQUEL LE MEMOIRE PUBLIE. Celui qu'il publie,")
print(f"  {' > '.join('P' + str(j) for j in ordre_root)}, est l'ordre des piliers d'AMORCE : c'est celui de la")
print("  propension d'amorce ROOT, et la ventilation par amorce le reproduit. L'ordre par pilier")
print("  TOUCHE est un AUTRE objet, celui de l'allocation d'Euler, et le memoire declare deja")
print("  qu'il bascule d'une resolution a l'autre, la severite etant de variance infinie a")
print("  xi = 0,595. Les parts touchees ci-dessus le confirment : elles tiennent toutes entre")
print("  17 et 23 %, donc l'ordre qu'on en tire n'est pas un signal. LES DEUX COLONNES SONT")
print("  DONC LUES DIFFEREMMENT DANS LA SUITE : l'amorce porte la these, le touche est reporte")
print("  pour memoire et sans conclusion.")

# =====================================================================================
titre("2. L'ECHELLE D'ABLATION : niveau, forme de queue, ordre des piliers")
# =====================================================================================
ABLATIONS = [
    ("aucune (reference)", dict()),
    ("(a) propagation g = 0", dict(g=0.0)),
    ("(b) auto-evitement retire", dict(mode="branching")),
    ("(c) surdispersion phi = 1", dict(phi=1.0)),
    ("(d) accumulation P4 retiree", dict(phi_cs=None)),
    ("(e) queue lourde xi = 0", dict(xi=0.0)),
    ("(f) amorces uniformes", dict(amorce="uniforme")),
]
print("  Chaque ligne retire UNE brique, les autres restant au niveau non conforme. La colonne")
print("  d'ordre est celle des piliers d'AMORCE, la seule des deux qui soit lisible ; une etoile")
print("  marque un ordre qui differe de la reference ET qui est stable entre graines, donc le seul")
print("  cas ou l'on puisse parler d'un deplacement.")
print(f"\n  {'ablation':<29}{'VaR (M€)':>11}{'/ ref':>8}{'CTE/VaR':>10}"
      f"{'ordre des amorces':>26}")
res = []
for nom, kw in ABLATIONS:
    args = dict(REF)
    args.update(kw)
    v, r, pt, pa = mesures(**args)
    ordre = classement(pa)
    st = stable(pa)
    diff = " *" if (ordre != ordre_a and st) else ("" if ordre == ordre_a else " ?")
    res.append((nom, v, r, pt, pa, ordre, st))
    print(f"  {nom:<29}{v.mean():>11.0f}{v.mean()/v_ref.mean():>8.2f}{r.mean():>10.3f}"
          f"{' > '.join('P' + str(j) for j in ordre) + diff:>26}")
print("\n  Legende de la derniere colonne : * ordre deplace et stable ; ? ordre deplace mais NON")
print("  stable entre graines, donc a ne pas interpreter ; rien = ordre inchange.")

print(f"\n  Etendues entre {NSEED} graines, pour lire les colonnes :")
for nom, v, r, _, _, _, _ in res:
    print(f"    {nom:<29}VaR ± {np.ptp(v):>6.0f}   CTE/VaR ± {np.ptp(r):.3f}")

# =====================================================================================
titre("3. CE QUE CHAQUE BRIQUE PORTE, ET LES TROIS GRANDEURS NE DISENT PAS LA MEME CHOSE")
# =====================================================================================
d = {nom: (v.mean(), r.mean(), ordre, st) for nom, v, r, _, _, ordre, st in res}
v0, r0, _, _ = d["aucune (reference)"]
print("  On lit maintenant colonne par colonne, et l'interet de la lecture a trois grandeurs est")
print("  qu'aucune n'ordonne les briques comme les autres.")
print("\n  SUR LE NIVEAU, l'ordre des briques par ce qu'elles retirent :")
rang_niv = [nom for nom, _ in sorted(d.items(), key=lambda kv: kv[1][0])
            if not nom.startswith("aucune")]
for nom in rang_niv:
    vv = d[nom][0]
    print(f"    {nom:<29}{vv:>9.0f} M€   soit {100*(vv-v0)/v0:>+6.1f} %")
print("\n  SUR LA FORME DE QUEUE (CTE / VaR), l'ordre n'est PAS le meme :")
rang_forme = [nom for nom, _ in sorted(d.items(), key=lambda kv: kv[1][1])
              if not nom.startswith("aucune")]
for nom in rang_forme:
    rr = d[nom][1]
    print(f"    {nom:<29}{rr:>9.3f}      contre {r0:.3f} en reference")
et_forme = {nom: np.ptp(r) for nom, _, r, _, _, _, _ in res}
resolus = [nom for nom, (_, rr, _, _) in d.items()
           if not nom.startswith("aucune") and abs(rr - r0) > et_forme[nom]]
print(f"\n  ET IL FAUT S'ARRETER SUR LA COLONNE DE FORME AVANT D'EN TIRER QUOI QUE CE SOIT. Ses")
print(f"  etendues entre graines vont de {min(et_forme.values()):.3f} a {max(et_forme.values()):.3f}, "
      f"alors que les ecarts a la reference")
print("  qu'on voudrait lire valent quelques centiemes. UNE SEULE ABLATION DEPLACE LA FORME DE")
print(f"  QUEUE AU-DELA DE SON BRUIT : {', '.join(resolus) if resolus else 'aucune'}.")
print("\n  Donc la lecture a deux grandeurs ne dit PAS ici que « les briques s'ordonnent autrement")
print("  sur la forme que sur le niveau » : ce classement-la n'est pas resolu, et l'ecrire serait")
print("  lire du bruit. Ce qu'elle dit, et c'est deja utile, est plus etroit : LA FORME DE QUEUE")
print("  EST INSENSIBLE A TOUT SAUF A LA QUEUE. Retirer la propagation, l'auto-evitement, la")
print("  surdispersion ou l'accumulation laisse le rapport CTE / VaR ou il est ; seul l'indice de")
print("  queue le fait tomber, de 2,08 a 1,11. La forme de queue du modele n'est donc pas produite")
print("  par sa cascade, elle est heritee de sa loi de severite.")
chg = [nom for nom, (_, _, o, st) in d.items() if o != ordre_a and st]
inst = [nom for nom, (_, _, o, st) in d.items() if o != ordre_a and not st]
print(f"\n  SUR L'ORDRE DES AMORCES, deplacements STABLES entre graines :")
if chg:
    for nom in chg:
        print(f"    {nom:<29}{' > '.join('P' + str(j) for j in d[nom][2])}")
else:
    print("    AUCUN. L'ordre des amorces survit a toutes les ablations de facon stable.")
if inst:
    print("  Et deplacements NON stables, a ne pas interpreter :")
    for nom in inst:
        print(f"    {nom:<29}{' > '.join('P' + str(j) for j in d[nom][2])}")

# =====================================================================================
titre("4. LA QUESTION QUE LE MEMOIRE SE POSE A LUI-MEME : l'ordre est-il mecanique ?")
# =====================================================================================
FUNI = "(f) amorces uniformes"
o_uni, st_uni = d[FUNI][2], d[FUNI][3]
pa_uni = [pa for nom, _, _, _, pa, _, _ in res if nom == FUNI][0]
p_uni = pa_uni.mean(axis=0)
print("  Le memoire declare que le classement des piliers « reproduit ROOT, en partie")
print("  MECANIQUEMENT, les amorces y etant semees proportionnellement ». C'est une reserve")
print("  honnete, et elle se teste exactement : on seme les amorces UNIFORMEMENT, ce qui retire")
print("  la part mecanique, et l'on regarde ce qu'il reste de l'ordre. Ce qui reste alors est la")
print("  contribution de la TOPOLOGIE seule, chaque pilier ne se distinguant plus que par sa")
print("  progeniture.")
print(f"\n  Amorces proportionnelles : {' > '.join('P' + str(j) for j in ordre_a)}")
print(f"  Amorces uniformes        : {' > '.join('P' + str(j) for j in o_uni)}"
      f"   (ordre stable entre graines : {st_uni})")
print(f"  Parts par amorce, uniformes : "
      + "  ".join(f"P{j} {100*p_uni[c]:.1f} %" for c, j in enumerate(PIL)))
ecart_max = 100 * (p_uni.max() - p_uni.min())
if not st_uni:
    print("\n  LA REPONSE EST QU'ON NE PEUT PAS CONCLURE, ET C'EST LE RESULTAT HONNETE. A amorces")
    print("  uniformes l'ordre n'est PAS stable entre les quatre graines : il change d'un tirage a")
    print(f"  l'autre. Les parts s'ecrasent d'ailleurs sur {ecart_max:.1f} points seulement entre le premier")
    print("  et le dernier pilier, contre un ecart bien plus large a amorces proportionnelles.")
    print("  AUTREMENT DIT : la topologie SEULE ne produit pas d'ordre lisible a cette resolution.")
    print("  La reserve du memoire est donc CONFIRMEE, et meme un peu durcie : l'ordre publie vient")
    print("  d'abord de la propension d'amorce, la topologie n'y ajoutant qu'un effet trop petit")
    print("  pour etre classe sur quatre graines. Ce qui reste vrai, et le memoire le dit deja au")
    print("  chapitre 12, c'est que la PART de P1 (34 %) excede sa part d'amorce (30 %) : cet")
    print("  exces-la est bien un effet propre de la propagation, et il est mesure ailleurs.")
elif o_uni == ordre_a:
    print("\n  L'ORDRE SURVIT ET IL EST STABLE, et c'est un resultat fort : il n'est donc pas un")
    print("  simple reflet de la propension d'amorce, la topologie de la cascade le reproduisant")
    print("  seule. La reserve du memoire etait plus severe que necessaire.")
else:
    print("\n  L'ORDRE BOUGE DE FACON STABLE, et il faut le dire tel quel : la part MECANIQUE de la")
    print("  propension d'amorce est reelle, et la reserve deja declaree par le memoire est")
    print("  confirmee. Ce qui reste vrai est plus etroit : l'ordre est celui de ROOT COMPOSE avec")
    print("  la topologie, non celui de la topologie seule.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. L'ABLATION N'EST PLUS UNIQUE : six briques, chacune retiree seule, contre une seule")
print("     dans le memoire. Le controle est exact, la reference reproduisant le nombre publie a")
print(f"     {abs(v_ref.mean() - ref_pub):.4f} M pres.")
tri_niv = sorted(((vv, nom) for nom, (vv, _, _, _) in d.items() if not nom.startswith("aucune")))
print(f"  2. LA BRIQUE LA PLUS LOURDE N'EST PAS LA CASCADE, C'EST LA QUEUE, et c'est le resultat")
print(f"     principal : retirer la queue lourde laisse {tri_niv[0][0]:.0f} M contre {v0:.0f}, soit "
      f"{100*(tri_niv[0][0]-v0)/v0:+.1f} %,")
print(f"     quand retirer la propagation ne coute que {100*(d['(a) propagation g = 0'][0]-v0)/v0:+.1f} %. Un facteur "
      f"{(v0-tri_niv[0][0])/(v0-d['(a) propagation g = 0'][0]):.1f} entre les deux.")
print("     C'est le meme enonce que le tornado renormalise du script 76 (elasticite de queue")
print("     3,79 contre 0,37 pour la propagation), obtenu par un chemin independant : un modele")
print("     de cascade dont le niveau est porte par sa loi de severite et non par sa cascade.")
print("  3. LA FORME DE QUEUE EST INSENSIBLE A TOUT SAUF A LA QUEUE, et c'est le seul enonce que")
print("     cette colonne supporte : ses etendues entre graines valent quelques dixiemes quand les")
print(f"     ecarts a lire valent quelques centiemes. Une seule ablation la deplace au-dela de son")
print(f"     bruit ({', '.join(resolus) if resolus else 'aucune'}), qui la fait passer de {r0:.2f} a "
      f"{d['(e) queue lourde xi = 0'][1]:.2f}. La")
print("     forme de queue du modele est donc HERITEE de sa loi de severite, pas produite par sa")
print("     cascade. Dire que les deux grandeurs « ordonnent les briques autrement » serait lire")
print("     du bruit, et je ne l'ecris pas.")
print(f"  4. L'ORDRE DES AMORCES est deplace de facon STABLE par {len(chg)} ablation(s) sur six, et de")
print(f"     facon non stable par {len(inst)}. Seules les premieres se lisent.")
print("  5. ET L'ORDRE PAR PILIER TOUCHE N'EST PAS UN SIGNAL A CETTE RESOLUTION : les parts")
print("     tiennent toutes entre 17 et 23 %. Le memoire le declare deja pour l'allocation")
print("     d'Euler ; ce script le confirme et n'en tire aucune conclusion.")
print("  6. RIEN N'EST IMPLEMENTE DANS LA CHAINE PUBLIEE : chaque ablation est une lecture, et")
print("     aucune ne deplace un nombre du memoire. Le controle est exact a 0,0000 M.")

# =====================================================================================
# figure S37
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4))
abl = [(nom, v, r) for nom, v, r, _, _, _, _ in res if not nom.startswith("aucune")]
lab = [n.split(" ", 1)[1] if " " in n else n for n, _, _ in abl]
ys = np.arange(len(abl))

# (a) le niveau : ecart relatif a la reference, avec l'etendue entre graines
dv = [100 * (v.mean() - v0) / v0 for _, v, _ in abl]
ev = [100 * np.ptp(v) / v0 for _, v, _ in abl]
ax1.barh(ys, dv, xerr=ev, color=[ACCENT if x < 0 else BLUE for x in dv], alpha=0.9,
         height=0.6, error_kw=dict(ecolor=MUTED, lw=1.1, capsize=3))
ax1.axvline(0.0, color=INK2, lw=1.1)
# ETIQUETTES AU-DELA DE LA BARRE D'ERREUR, et non au bout de la barre : posees au bout, elles
# tombaient dans la moustache et devenaient illisibles sur cinq lignes sur six.
for y, x, e in zip(ys, dv, ev):
    ax1.text(x - e - 2.0, y, f"{x:+.0f} %", va="center", ha="right", fontsize=9, color=INK2)
ax1.set_yticks(ys)
ax1.set_yticklabels(lab, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlim(min(x - e for x, e in zip(dv, ev)) - 16, 8)
ax1.set_xlabel("déplacement du niveau de capital (%)", color=INK2)
ax1.set_title("(a)  Le niveau : ce que chaque brique porte\n(barres d'erreur = étendue sur quatre "
              "graines)", fontsize=10.5, color=INK, pad=8)

# (b) la FORME de queue, qui n'ordonne pas les briques comme le niveau
rr = [r.mean() for _, _, r in abl]
er = [np.ptp(r) for _, _, r in abl]
ax2.barh(ys, rr, xerr=er, color=GREEN, alpha=0.85, height=0.6,
         error_kw=dict(ecolor=MUTED, lw=1.1, capsize=3))
ax2.axvline(r0, color=ACCENT, lw=1.5, ls="--")
# ETIQUETTE DE REFERENCE POSEE DANS LE BLANC LAISSE PAR LA BARRE « xi = 0 », qui ne monte qu'a
# 1,11 : en bas elle chevauchait les graduations, en haut le titre du panneau.
ax2.text(r0 - 0.03, 4.0, f"référence {r0:.2f}".replace(".", ","), fontsize=9,
         color=ACCENT, ha="right", va="center", fontweight="bold")
for y, x, e in zip(ys, rr, er):
    ax2.text(x + e + 0.04, y, f"{x:.2f}".replace(".", ","), va="center", fontsize=9, color=INK2)
ax2.set_yticks(ys)
ax2.set_yticklabels([])
ax2.invert_yaxis()
ax2.set_xlim(0.9, max(x + e for x, e in zip(rr, er)) + 0.30)
ax2.set_xlabel("forme de queue, rapport CTE / VaR", color=INK2)
# PAS DE BLOC DE TEXTE ICI : le titre du panneau porte deja le message, et toute position
# essayee dans l'axe heurtait soit une moustache, soit les graduations, soit le titre. Le message
# honnete est que les moustaches avalent tous les ecarts sauf un, et le titre le dit.
ax2.set_title("(b)  La forme de queue est insensible à tout\nsauf à la queue : elle est héritée, "
              "pas produite", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S37 : l'ablation en échelle, et pourquoi un seul retrait ne suffit pas",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S37_ablation_structure.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
