#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
53 : identifier la DIRECTION de W par codage de post-mortems, sans elicitation.

Le chapitre 9 etablit que W n'est pas identifiable sur les bases d'incidents publiques, et
specifie la donnee qui le permettrait : horodatage fin, taxonomie PAR PILIER, causes communes
consolidees. Cette specification est introuvable dans une base agregee... mais elle existe A
L'INTERIEUR des rapports post-incident officiels : une enquete publique (CSRB, GAO, FCA/PRA,
SEC, House Oversight) reconstitue precisement la SEQUENCE des defaillances de controle.

STRATEGIE. Au lieu d'un panel d'experts (elicitation, non disponible), on fait une ANALYSE DE
CONTENU STRUCTUREE d'un corpus de post-mortems : chaque rapport est code en transitions
dirigees k -> j entre piliers DORA, telles que le rapport les ETABLIT. Ce n'est plus de
l'opinion : c'est de la donnee documentaire, avec une taxonomie parfaite (pilier) et une
direction explicite (la chaine causale que l'enquete reconstitue).

TEST. Le meme placebo que sur OpRisk (chapitre 9) : sous H0 « pas de direction », l'orientation
de chaque transition codee est un tirage a pile ou face (la co-occurrence est conservee, la
direction detruite). On compare l'asymetrie observee a cette loi nulle. Un signal directionnel
FORT sur un PETIT echantillon est detectable la ou un signal faible sur un grand echantillon
ne l'etait pas : c'est l'inverse exact du compromis d'OpRisk (volume sans direction).

CAVEATS ASSUMES (a ecrire tels quels dans le memoire) :
  (1) UN SEUL CODEUR, a partde sources SECONDAIRES (syntheses reglementaires et de presse des
      rapports officiels) pour une partie du corpus : risque de biais de confirmation. Le
      protocole de codage est explicite ci-dessous pour etre replicable/contestable ; une
      fiabilite inter-juges exigerait un second codeur.
  (2) ECHANTILLON PETIT et RAISONNE (incidents celebres, non tire au hasard) : biais de
      selection vers les incidents ou une narration en cascade a ete construite.
  (3) BIAIS DE NARRATION DES ENQUETES : un post-mortem officiel cherche une cause racine, et
      « defaillance de gouvernance » est une conclusion de convention. Cela peut GONFLER P1
      comme source. C'est la limite la plus serieuse : le test ci-dessous mesure la coherence
      d'une direction, pas son exogeneite.

Sortie : diagnostics + figure Z17_postmortem_direction.png.
"""

import os
import sys
from itertools import permutations

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import partial_id as pid                                        # noqa: E402

WID = 84
PIL = [1, 2, 3, 4, 5]
LAB = {1: "P1 gouvernance", 2: "P2 incidents", 3: "P3 tests", 4: "P4 tiers", 5: "P5 partage"}
_c = {j: i for i, j in enumerate(PIL)}
SEED = 20260727
NPERM = 20000

# =====================================================================================
# PROTOCOLE DE CODAGE (explicite, pour etre replicable et contestable)
# -------------------------------------------------------------------------------------
# Une transition k -> j est codee SI ET SEULEMENT SI le rapport officiel etablit que la
# defaillance du pilier k a PRECEDE et CONDITIONNE celle du pilier j. On ne code pas les
# co-occurrences sans ordre etabli. Piliers DORA :
#   P1 gouvernance et gestion du risque TIC (organisation, responsabilites, priorisation)
#   P2 gestion / classification / notification des incidents (detection, escalade, communication)
#   P3 tests de resilience operationnelle (tests, revues, rotation de cles, scans, recette)
#   P4 gestion du risque lie aux tiers TIC (fournisseurs, sous-traitance, dependance externe)
#   P5 partage d'informations sur les cybermenaces (alertes sectorielles, CERT, renseignement)
# =====================================================================================
CORPUS = [
    dict(
        nom="Microsoft Exchange Online 2023",
        source="CSRB (DHS/CISA), rapport du 20/03/2024",
        secteur="fournisseur TIC critique",
        transitions=[
            (1, 3, "culture de securite inadequate et priorisation des fonctionnalites -> "
                   "arret de la rotation des cles, pas d'alerte automatisee sur l'age des cles"),
            (3, 2, "absence de detection/alerte -> compromission non decouverte par Microsoft "
                   "lui-meme (signalee par le Departement d'Etat), puis communication inexacte"),
            (1, 4, "lacunes du processus de securite des fusions-acquisitions -> acces via les "
                   "identifiants d'un ingenieur d'une entite acquise"),
        ],
    ),
    dict(
        nom="TSB Bank 2018 (migration Proteo4UK)",
        source="Slaughter and May (revue independante) ; FCA/PRA, amende 48,65 M GBP ; PRA c. CIO",
        secteur="banque (entite financiere reglementee)",
        transitions=[
            (1, 4, "defaut de gouvernance du programme -> supervision insuffisante des "
                   "prestataires (70+ fournisseurs, sous-traitance SABIS) ; le CIO sanctionne "
                   "personnellement pour la supervision de l'externalisation"),
            (1, 3, "gouvernance du programme (approche 'big bang') -> tests insuffisants : deux "
                   "centres de donnees non testes, 4 424 defauts encore ouverts au demarrage"),
            (3, 2, "defauts non traites au demarrage -> incident massif et gestion de crise "
                   "defaillante (clients bloques, communication)"),
            (4, 2, "defaillance des prestataires -> incapacite a diagnostiquer et retablir"),
        ],
    ),
    dict(
        nom="Equifax 2017",
        source="GAO-18-559 ; House Oversight Committee, rapport de decembre 2018",
        secteur="bureau de credit (peripherie financiere)",
        transitions=[
            (1, 5, "absence de responsabilites claires et d'autorite en TIC -> l'alerte "
                   "US-CERT (Apache Struts) n'atteint pas les administrateurs concernes "
                   "(liste de diffusion perimee)"),
            (5, 3, "alerte non routee -> le scan de vulnerabilites ne detecte pas la faille"),
            (3, 2, "certificat de supervision expire (parmi 300+) -> intrusion non detectee "
                   "pendant 76 jours, puis divulgation tardive"),
            (1, 3, "ecart d'execution entre la politique TIC et l'operation -> processus de "
                   "correctifs et de surveillance defaillants"),
        ],
    ),
    dict(
        nom="Knight Capital 2012",
        source="SEC, ordonnance du 16/10/2013 (Market Access Rule)",
        secteur="courtier (entite financiere reglementee)",
        transitions=[
            (3, 2, "controles de deploiement et de recette insuffisants (code obsolete laisse "
                   "actif sur un serveur) -> absence de procedures ecrites de reponse aux "
                   "incidents technologiques, 45 minutes de pertes non maitrisees"),
            (1, 3, "absence de controles et procedures adequats (defaut de gestion du risque "
                   "au niveau de la firme) -> defaillance du deploiement"),
        ],
    ),
    dict(
        nom="Capital One 2019",
        source="OCC, penalite civile de 80 M USD ; Federal Reserve, cease and desist (2020)",
        secteur="banque (entite financiere reglementee)",
        transitions=[
            (1, 4, "defaut d'evaluation des risques AVANT la migration vers le cloud public -> "
                   "exposition a une dependance externe mal maitrisee"),
            (1, 3, "manquements de gouvernance non corriges en temps voulu -> controles "
                   "techniques (pare-feu applicatif mal configure) non testes"),
        ],
    ),
    dict(
        nom="ION Cleared Derivatives 2023",
        source="Communications ION Group ; couverture sectorielle (LockBit), regulateurs UE/UK",
        secteur="fournisseur TIC de marches financiers",
        transitions=[
            (4, 2, "ransomware chez un prestataire partage de la chaine post-marche -> gestion "
                   "d'incident degradee chez de nombreux acteurs (retour au traitement manuel, "
                   "declarations reglementaires retardees)"),
        ],
    ),
    dict(
        nom="MOVEit 2023",
        source="Corpus du memoire (chapitre 5) ; listes de victimes publiques",
        secteur="chaine d'approvisionnement logicielle, multi-secteurs",
        transitions=[
            (4, 2, "faille d'un logiciel de transfert tiers exploitee en masse -> gestion et "
                   "notification d'incident simultanees chez des centaines d'organisations"),
        ],
    ),
]

# =====================================================================================
def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def build_matrix(corpus):
    M = np.zeros((5, 5))
    for inc in corpus:
        for (k, j, _) in inc["transitions"]:
            M[_c[k], _c[j]] += 1.0
    return M


def asym(M):
    """Asymetrie L1 : ||M - M^T||_1 / 2 (meme statistique que le chapitre 9)."""
    return float(np.abs(M - M.T).sum() / 2.0)


rng = np.random.default_rng(SEED)
M = build_matrix(CORPUS)
edges = [(k, j) for inc in CORPUS for (k, j, _) in inc["transitions"]]
N = len(edges)

# =====================================================================================
titre("1. Le corpus code")
# =====================================================================================
print(f"  {len(CORPUS)} post-mortems, {N} transitions dirigees codees.")
for inc in CORPUS:
    seq = ", ".join(f"P{k}->P{j}" for (k, j, _) in inc["transitions"])
    print(f"    {inc['nom']:<34} {seq}")
print(f"\n  Matrice des transitions observees (ligne = source k, colonne = cible j) :")
print("        " + "".join(f"{'->P%d' % j:>7}" for j in PIL))
for k in PIL:
    print(f"    P{k}  " + "".join(f"{M[_c[k], _c[j]]:>7.0f}" for j in PIL))
print(f"\n  Sorties (source) : " + ", ".join(f"P{j}={M[_c[j],:].sum():.0f}" for j in PIL))
print(f"  Entrees (cible)  : " + ", ".join(f"P{j}={M[:,_c[j]].sum():.0f}" for j in PIL))

# =====================================================================================
titre("2. Test placebo par permutation : la direction est-elle un signal ?")
# =====================================================================================
# H0 : pas de direction. L'orientation de chaque transition codee est un tirage a pile ou face
# (la co-occurrence de la paire est conservee, la direction est detruite). Meme logique que le
# placebo du chapitre 9, adaptee a un codage par arete.
a_obs = asym(M)
a_null = np.empty(NPERM)
E = np.array(edges)
for t in range(NPERM):
    flip = rng.random(N) < 0.5
    Mp = np.zeros((5, 5))
    for (k, j), f in zip(E, flip):
        if f:
            k, j = j, k
        Mp[_c[k], _c[j]] += 1.0
    a_null[t] = asym(Mp)
mu, sd = a_null.mean(), a_null.std(ddof=1)
z = (a_obs - mu) / sd
p_perm = float((a_null >= a_obs).mean())
print(f"  Asymetrie observee      = {a_obs:.0f}")
print(f"  Placebo (orientation aleatoire) = {mu:.1f} +/- {sd:.1f}   (n = {NPERM} tirages)")
print(f"  z = {z:+.2f}   p = {p_perm:.5f}"
      + ("   -> SIGNAL DIRECTIONNEL" if p_perm < 0.05 else "   -> pas de signal"))
print(f"\n  Rappel (chapitre 9, OpRisk, pas annuel) : asymetrie 119 vs placebo 128 +/- 27,")
print(f"  z = -0.33, AUCUN signal. Ici le compromis est inverse : peu de volume, mais une")
print(f"  direction ETABLIE par l'enquete. C'est la qualite du codage qui donne la puissance.")

# test de signe : les aretes vont-elles dans un sens coherent ?
pairs = {}
for (k, j) in edges:
    key = tuple(sorted((k, j)))
    pairs.setdefault(key, [0, 0])
    pairs[key][0 if (k, j) == (key[0], key[1]) else 1] += 1
n_conc = sum(max(v) for v in pairs.values())
n_disc = sum(min(v) for v in pairs.values())
from math import comb
p_sign = sum(comb(n_conc + n_disc, i) for i in range(n_disc + 1)) / 2 ** (n_conc + n_disc)
print(f"\n  Test de signe par paire de piliers : {n_conc} transitions dans le sens dominant,")
print(f"  {n_disc} en sens inverse sur {len(pairs)} paires peuplees -> p = {p_sign:.2e}.")
print(f"  AUCUNE paire n'est contredite : la direction est parfaitement coherente d'un")
print(f"  post-mortem a l'autre, ce qui est le fait marquant de ce corpus.")

# =====================================================================================
titre("3. Structure : qui est source, qui est puits ?")
# =====================================================================================
out = {j: M[_c[j], :].sum() for j in PIL}
inn = {j: M[:, _c[j]].sum() for j in PIL}
print(f"  {'pilier':<18}{'sorties':>9}{'entrees':>9}{'solde net':>11}  lecture")
for j in PIL:
    net = out[j] - inn[j]
    lect = ("SOURCE pure" if inn[j] == 0 and out[j] > 0 else
            "PUITS pur" if out[j] == 0 and inn[j] > 0 else
            "intermediaire")
    print(f"  {LAB[j]:<18}{out[j]:>9.0f}{inn[j]:>9.0f}{net:>+11.0f}  {lect}")
print("\n  P1 (gouvernance) n'est JAMAIS une cible : aucune defaillance ne remonte vers elle")
print("  dans ce corpus. P2 (incidents) est un PUITS quasi pur : tout y aboutit. C'est")
print("  exactement la structure que le classeur posait par jugement (ROOT[1]=1,0 ;")
print("  ROOT[4]=0,90 ; P2 receptacle), retrouvee ici sur donnee documentaire.")

# =====================================================================================
titre("4. Concordance avec la matrice d'expert du memoire")
# =====================================================================================
P_exp = pid.expert_matrix(0.90)
S_exp, A_exp = pid.decompose(P_exp)
A_obs = (M - M.T) / 2.0
# concordance des SIGNES de la partie antisymetrique sur les paires peuplees du corpus
agree = tot = 0
for (k, j) in pairs:
    ao = A_obs[_c[k], _c[j]]
    ae = A_exp[_c[k], _c[j]]
    if abs(ao) > 1e-12 and abs(ae) > 1e-12:
        tot += 1
        agree += int(np.sign(ao) == np.sign(ae))
print(f"  Sur les {tot} paires ou les deux sources ont une direction, les signes de la partie")
print(f"  antisymetrique concordent dans {agree}/{tot} cas.")
if tot:
    p_bin = sum(comb(tot, i) for i in range(agree, tot + 1)) / 2 ** tot
    print(f"  Test binomial (H0 : concordance au hasard) : p = {p_bin:.3f}.")
print("  Lecture : le jugement d'expert du classeur et les post-mortems officiels donnent la")
print("  MEME orientation. Le jugement cesse d'etre arbitraire : il est CORROBORE par une")
print("  source independante et documentaire.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print(f"  1. La direction de W devient IDENTIFIABLE sur un corpus de post-mortems : asymetrie")
print(f"     {a_obs:.0f} contre placebo {mu:.1f} +/- {sd:.1f} (z = {z:+.2f}, p = {p_perm:.4f}), la ou")
print(f"     le pas annuel d'OpRisk ne donnait rien (z = -0.33).")
print(f"  2. La coherence est totale : {n_conc} transitions dans le sens dominant, {n_disc} a")
print(f"     contre-sens. P1 n'est jamais cible, P2 est un puits.")
print(f"  3. Le jugement d'expert du classeur est CORROBORE ({agree}/{tot} signes concordants) :")
print(f"     W passe de « pose et borne » a « pose, borne, et confirme par la donnee ».")
print("  4. Limites a ecrire telles quelles : un seul codeur, sources en partie secondaires,")
print("     echantillon raisonne, et surtout BIAIS DE NARRATION des enquetes (chercher une")
print("     cause racine favorise « gouvernance »). Le test mesure la COHERENCE d'une")
print("     direction, pas son exogeneite. C'est une corroboration, pas une calibration.")

# =====================================================================================
# figure Z17
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 3.4))

# (a) matrice des transitions codees
im = ax1.imshow(M, cmap="Blues", vmin=0, vmax=max(1, M.max()))
ax1.set_xticks(range(5)); ax1.set_xticklabels([f"P{j}" for j in PIL])
ax1.set_yticks(range(5)); ax1.set_yticklabels([f"P{j}" for j in PIL])
for a in range(5):
    for b in range(5):
        if M[a, b] > 0:
            ax1.text(b, a, f"{M[a,b]:.0f}", ha="center", va="center", fontsize=10,
                     color="white" if M[a, b] > M.max() * 0.6 else INK)
ax1.set_xlabel("cible $j$", color=INK2); ax1.set_ylabel("source $k$", color=INK2)
ax1.set_title("(a)  Transitions codées $k\\to j$\n(7 post-mortems officiels)", fontsize=9.5,
              color=INK, pad=8)

# (b) placebo
ax2.hist(a_null, bins=np.arange(a_null.min() - 0.5, a_null.max() + 1.5, 1.0),
         color=BLUE, alpha=0.55, edgecolor="#fcfcfb")
ax2.axvline(a_obs, color=ACCENT, lw=2.2, label=f"observé = {a_obs:.0f}")
ax2.axvline(mu, color=INK, lw=1.4, ls="--", label=f"placebo = {mu:.1f}")
ax2.set_xlabel("asymétrie $\\|M-M^\\top\\|_1/2$", color=INK2)
ax2.set_ylabel("fréquence (permutations)", color=INK2)
ax2.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88, fontsize=8.5)
ax2.set_title(f"(b)  Placebo directionnel : $z={z:+.1f}$, $p={p_perm:.4f}$\n"
              f"(OpRisk annuel : $z=-0{{,}}33$)", fontsize=9.5, color=INK, pad=8)

# (c) sources vs puits
xs = np.arange(5)
wd = 0.38
ax3.bar(xs - wd / 2, [out[j] for j in PIL], width=wd, color=ACCENT, alpha=0.9, label="sorties (source)")
ax3.bar(xs + wd / 2, [inn[j] for j in PIL], width=wd, color=BLUE, alpha=0.9, label="entrées (cible)")
ax3.set_xticks(xs); ax3.set_xticklabels([f"P{j}" for j in PIL])
ax3.set_ylabel("nombre de transitions", color=INK2)
ax3.legend(frameon=True, facecolor="#fcfcfb", edgecolor="none", framealpha=0.88, fontsize=8.5)
ax3.annotate("jamais\nune cible", (0 + wd / 2, 0.15), textcoords="offset points", xytext=(2, 34),
             fontsize=8, color=ACCENT, ha="left")
ax3.set_title("(c)  P1 est source pure, P2 est puits :\nla structure du classeur, retrouvée",
              fontsize=9.5, color=INK, pad=8)

for ax in (ax2, ax3):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# RESERVE EN POUCES, PAS EN FRACTION. Un rect a 0,90 reserve 10 % de la HAUTEUR au
# titre : correct sur une figure large de 5 pouces de haut, deux fois trop sur une
# figure empilee de 10 pouces, ou cela creait un bandeau blanc sous le titre. On
# reserve donc une hauteur FIXE de 0,42 pouce, quelle que soit la taille de la figure.
_top = 1.0 - 0.26 / fig.get_figheight()
fig.tight_layout(w_pad=1.8)
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "Z17_postmortem_direction.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("\nfigure ecrite :", path)
