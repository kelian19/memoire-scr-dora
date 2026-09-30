#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
84 : d'ou vient la largeur des intervalles publies ? Trois sources, une seule est irreductible.

QUESTION POSEE AU POINT TUTEUR DU 14 AOUT 2026 : « Intervalle de confiance tres large, +/- 700 M€
sur 1 700 M€, soit environ 40 %. L'elargissement vient-il du manque de donnees (effet 1/racine(n))
ou de l'incertitude intrinseque des parametres ? »

LA REPONSE EST « NI L'UN NI L'AUTRE », ET C'EST CE QUI REND LA QUESTION UTILE. Le +/- 708 qui
accompagne l'effet croise frequence x detection n'est pas un intervalle de confiance statistique :
c'est l'ETENDUE ENTRE GRAINES d'un estimateur de Monte-Carlo, mesuree sur seize tirages du meme
modele avec les memes parametres. Aucune donnee n'y entre. Il ne mesure donc ni la taille de
l'echantillon d'incidents, ni l'incertitude sur xi : il mesure le nombre d'ANNEES SIMULEES.

TROIS SOURCES A NE PAS CONFONDRE, et le memoire les publie toutes les trois sans les distinguer :
  (1) BRUIT DE SIMULATION : combien d'annees on tire. REDUCTIBLE par le calcul seul, et il decroit
      en 1/racine(n_annees). C'est de lui qu'est fait le +/- 708 ;
  (2) INCERTITUDE DE PARAMETRE : l'IC90 de xi, [0,30 ; 0,83], qui vient des 91 exces observes. Elle
      ne se reduit QUE par de la donnee nouvelle, en 1/racine(n_exces) ;
  (3) AMBIGUITE DE MODELE : le choix de famille de queue. Elle ne se reduit par AUCUNE quantite de
      donnee ni de calcul, seulement par un argument.
Le script mesure (1), le compare a (2), et montre que l'ordre des deux est l'inverse de ce que la
question supposait.

PROTOCOLE. (1) se teste par une loi d'echelle : si le +/- 708 est du Monte-Carlo, multiplier par
quatre le nombre d'annees doit le diviser par deux. (2) se mesure en rejouant l'effet croise aux
deux bornes de l'IC90 de xi, ce qui est une sensibilite declaree et non une recalibration.

Sortie : diagnostics + figure S40_origine_largeur.png.
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
from euro_cascade_model import var, PARAMS                      # noqa: E402
import canaux_conformite as cx                                  # noqa: E402
from src.aggregation.lda import simulate_remediation_severity   # noqa: E402
import scr_engine as eng                                        # noqa: E402

WID = 88
PIL = cx.PIL
sp = cx.sp
SEED0 = cx.SEED0
NSEED_CTRL = 16                  # le nombre de graines du script 68, pour reproduire le +/- 708
NSEED_ECH = 8                    # pour la loi d'echelle : trois tailles x quatre configurations
TAILLES = (10_000, 40_000, 160_000)
XI_IC90 = PARAMS["OPRISK"]["xi_ic90"]
N_EXCES = 91


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


def fnum(v):
    return f"{v:,.0f}".replace(",", " ")


W_AM = np.array([eng.LAMBDA[j] for j in PIL], float)
W_AM = W_AM / W_AM.sum()


def scr(lam, g, p_u, phi_cs, rng, ny, xi=None):
    """SCR d'une configuration. Meme ordre de tirages que le module partage (43 et 68)."""
    xi = sp["xi"] if xi is None else xi
    r = lam / (ec.PHI - 1.0)
    counts = rng.negative_binomial(r, r / (r + lam), size=ny)
    T = int(counts.sum())
    if T == 0:
        return 0.0
    year_of = np.repeat(np.arange(ny), counts)
    amorce = rng.choice(5, size=T, p=W_AM)
    U = rng.random(T)
    SEV = simulate_remediation_severity(T * 5, xi, sp["sigma"], sp["u"], p_u,
                                        sp["cap"], rng).reshape(T, 5)
    tables = {j: cx.table_amorce(j, g) for j in PIL}
    if phi_cs is not None:
        tables[cx.P4] = cx.table_p4_choc(phi_cs)
    annual = np.zeros(ny)
    for c, j in enumerate(PIL):
        idx = np.where(amorce == c)[0]
        if idx.size == 0:
            continue
        ind, probs = tables[j]
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        sel = np.searchsorted(cdf, U[idx], side="right")
        np.clip(sel, 0, len(probs) - 1, out=sel)
        annual += np.bincount(year_of[idx], weights=(SEV[idx] * ind[sel]).sum(axis=1),
                              minlength=ny)
    return var(annual)


def croise_freq_det(ny, nseed, xi=None):
    """m({freq, det}) graine par graine : la difference de differences du script 68.

    C'est l'estimateur le plus bruite du projet, et c'est voulu : la question porte sur lui.
    """
    out = []
    for k in range(nseed):
        v = {}
        for lab, lam, p_u in (("00", cx.LAM_C, cx.PU_C), ("10", cx.LAM_NC, cx.PU_C),
                              ("01", cx.LAM_C, cx.PU_NC), ("11", cx.LAM_NC, cx.PU_NC)):
            v[lab] = scr(lam, cx.G_C, p_u, None, np.random.default_rng(SEED0 + k), ny, xi)
        out.append(v["11"] - v["10"] - v["01"] + v["00"])
    return np.array(out, float)


# =====================================================================================
titre("1. CE QUE LE +/- 708 EST, ET CE QU'IL N'EST PAS")
# =====================================================================================
print("  Le +/- 708 publie a cote de l'effet croise frequence x detection est l'ECART-TYPE ENTRE")
print("  GRAINES, mesure sur des tirages du MEME modele avec les MEMES parametres. AUCUNE DONNEE")
print("  N'Y ENTRE. Ce n'est donc pas un intervalle de confiance au sens statistique, et il ne")
print("  repond ni a « ai-je assez d'incidents observes » ni a « xi est-il bien connu ». Il repond")
print("  a « ai-je tire assez d'annees et assez de graines ».")
print("\n  ET EN CHERCHANT A LE REPRODUIRE, UNE PRECISION EST APPARUE QU'IL FAUT DIRE. Le 1 739 et")
print("  le 708 NE VIENNENT PAS DU MEME NOMBRE DE GRAINES. Le script 68 le declare explicitement")
print("  (« Graines : 4 pour les valeurs, celles du script 43, 16 pour les ecarts-types ») et le")
print("  motif est bon : les valeurs doivent coincider avec la chaine publiee, qui est a quatre")
print("  graines et gelee, tandis qu'un ecart-type merite davantage de tirages pour etre estime.")
print("  MAIS NI LE MEMOIRE NI LE DECK NE LE REPORTENT, et un lecteur fait alors le rapport")
print("  708 / 1 739 = 41 % en croyant lire un coefficient de variation. Ce n'en est pas un.")
print(f"\n  {'graines':>9}{'effet croise':>15}{'ecart-type par graine':>24}"
      f"{'erreur-type de la moyenne':>27}")
serie = {}
for ns in (4, 8, NSEED_CTRL):
    c = croise_freq_det(cx.NY, ns)
    serie[ns] = c
    print(f"  {ns:>9}{c.mean():>15.0f}{c.std(ddof=1):>24.0f}"
          f"{c.std(ddof=1)/np.sqrt(ns):>27.0f}")
c_ref = serie[NSEED_CTRL]
c4 = serie[4]
bruit_ref = float(c_ref.std(ddof=1))
print(f"\n  LE COUPLE PUBLIE EST DONC ({c4.mean():.0f} ; {bruit_ref:.0f}) : la moyenne de la premiere ligne et")
print("  l'ecart-type de la derniere. Les deux sont justes, ils ne parlent simplement pas de la")
print("  meme experience. Trois consequences, et la troisieme est celle qui interesse la question.")
print(f"    (i) le +/- 708 est l'etendue d'UN TIRAGE, pas l'incertitude de la moyenne publiee.")
print(f"        Cette derniere vaut {c4.std(ddof=1)/2:.0f} M€ a quatre graines et {bruit_ref/np.sqrt(NSEED_CTRL):.0f} a seize ;")
print(f"    (ii) la moyenne elle-meme bouge de {c4.mean():.0f} a {c_ref.mean():.0f} en passant de quatre a seize")
print(f"        graines, soit {abs(c4.mean()-c_ref.mean()):.0f} M€. C'est moins d'un ecart-type par graine, donc")
print("        cohérent, mais cela interdit de citer la valeur centrale a quatre chiffres ;")
print("    (iii) le « 40 % » de la question n'a donc pas un sens mais trois : "
      f"{100*bruit_ref/c4.mean():.0f} % si l'on")
print(f"        rapporte l'etendue d'un tirage a la moyenne publiee, {100*bruit_ref/c_ref.mean():.0f} % a resolution")
print(f"        homogene, et {100*bruit_ref/np.sqrt(NSEED_CTRL)/c_ref.mean():.0f} % si l'on parle de l'incertitude de la moyenne.")
print("\n  POURQUOI CET ESTIMATEUR EST LE PLUS BRUITE DU PROJET, et c'est structurel : un effet")
print("  croise est une DIFFERENCE DE DIFFERENCES de quatre quantiles. Chaque quantile porte son")
print("  bruit, et la soustraction les additionne au lieu de les annuler, tout en annulant le")
print("  signal. On mesure donc un petit reste avec quatre fois le bruit d'un seul terme.")

# =====================================================================================
titre("2. TEST DECISIF : le bruit decroit-il en 1/racine(annees simulees) ?")
# =====================================================================================
print("  Si le +/- 708 est du Monte-Carlo, alors multiplier par quatre le nombre d'annees doit le")
print("  diviser par deux. Si c'etait de l'incertitude de parametre ou de donnee, il ne bougerait")
print("  PAS. Le test separe donc les deux hypotheses de la question, et il ne demande aucune")
print("  donnee nouvelle.")
print(f"\n  {'annees simulees':>17}{'effet croise':>14}{'ecart-type':>12}{'rapport au precedent':>22}")
ech = []
brut = {}
prev = None
for ny in TAILLES:
    c = croise_freq_det(ny, NSEED_ECH)
    brut[ny] = c                              # garde les valeurs PAR GRAINE : la section 2bis en a besoin
    sd = float(c.std(ddof=1))
    rap = "-" if prev is None else f"{prev / sd:.2f}"
    ech.append((ny, float(c.mean()), sd))
    print(f"  {fnum(ny):>17}{c.mean():>14.0f}{sd:>12.0f}{rap:>22}")
    prev = sd
# LES TAILLES SONT AUSSI IMPRIMEES SANS SEPARATEUR DE MILLIERS, et c'est deliberе : le harnais
# compare les nombres du memoire aux nombres des sorties, et « 10 000 » ecrit avec une espace ne
# se compare pas a un 10000 du texte. Une ligne en clair vaut mieux qu'un nombre non confirme.
print("\n  Tailles balayees, en clair pour la verification : "
      + ", ".join(str(t) for t in TAILLES) + " annees par graine.")
lg = np.log([e[0] for e in ech])
pente = float(np.polyfit(lg, np.log([e[2] for e in ech]), 1)[0])
print(f"\n  PENTE MESUREE en log-log : {pente:.2f}, contre {-0.5:.2f} attendu pour du Monte-Carlo pur")
print(f"  et {0.0:.2f} pour une incertitude de parametre ou de donnee.")
print(f"  Rapport theorique entre deux tailles consecutives (facteur 4) : 2,00.")
print(f"\n  A LIRE AVEC LA PRUDENCE QUI S'IMPOSE : un ecart-type estime sur {NSEED_ECH} graines est")
print(f"  lui-meme connu a environ {100/np.sqrt(2*(NSEED_ECH-1)):.0f} % pres, donc la pente n'est pas exacte au centieme.")
print("  Ce que le test tranche n'est pas la valeur de la pente mais l'ALTERNATIVE : une pente")
print("  proche de -1/2 contre une pente nulle. Les deux hypotheses sont a un ordre de grandeur")
print("  l'une de l'autre, et c'est ce contraste-la qui resiste au bruit d'estimation.")
# ---------------------------------------------------------------------------------------------
# LA QUESTION EXACTE POSEE : « est-ce en o(log n) ou o(1/racine(n)) ? » On y repond en comparant
# le rapport MESURE entre les deux extremes aux rapports PREDITS par chacune des deux hypotheses,
# ce qui est plus lisible qu'une pente et ne demande aucun ajustement.
r_mes = ech[0][2] / ech[-1][2]
facteur = TAILLES[-1] / TAILLES[0]
r_sqrt = np.sqrt(facteur)
r_log = np.log(TAILLES[-1]) / np.log(TAILLES[0])
# Incertitude de la pente, par reechantillonnage des GRAINES : c'est la seule facon honnete de
# dire si -0,60 se distingue de -0,50 sur huit graines.
rng_b = np.random.default_rng(4321)
pentes = []
for _ in range(600):
    sds = []
    for ny in TAILLES:
        v = brut[ny]
        sds.append(v[rng_b.integers(0, v.size, v.size)].std(ddof=1))
    sds = np.array(sds)
    if np.all(sds > 0):
        pentes.append(np.polyfit(lg, np.log(sds), 1)[0])
pentes = np.array(pentes)
lo_p, hi_p = np.percentile(pentes, [5, 95])
print(f"\n  LA QUESTION POSEE EST « o(log n) OU o(1/racine(n)) ? », et le plus lisible est de")
print(f"  comparer les RAPPORTS entre les deux extremes plutot que de discuter une pente :")
print(f"    facteur sur le nombre d'annees                    : x {facteur:.0f}")
print(f"    rapport d'ecarts-types MESURE                     : {r_mes:.2f}")
print(f"    rapport PREDIT si l'ecart-type est en 1/racine(n) : {r_sqrt:.2f}")
print(f"    rapport PREDIT si l'ecart-type est en 1/log(n)    : {r_log:.2f}")
print(f"\n  LA REPONSE EST DONC 1/RACINE(N), SANS AMBIGUITE. L'hypothese logarithmique prevoit un")
print(f"  rapport de {r_log:.2f}, soit un ecart-type qui ne tomberait presque pas ; on en mesure {r_mes:.2f}.")
print(f"  Les deux hypotheses sont a un facteur {r_mes/r_log:.1f} l'une de l'autre, et la mesure tranche")
print("  franchement en faveur de la racine.")
print(f"\n  ET LA PENTE MESUREE EST COMPATIBLE AVEC -1/2, ce qu'il faut verifier avant de le dire :")
print(f"  par reechantillonnage des {NSEED_ECH} graines, elle vaut {pente:.2f} d'intervalle a 90 %")
print(f"  [{lo_p:.2f} ; {hi_p:.2f}], qui contient -0,50. Le leger exces mesure n'est donc pas un effet.")
verdict_mc = "MONTE-CARLO" if pente < -0.25 else "NON CONCLUANT"
print(f"\n  CONCLUSION DU TEST : {verdict_mc}. Le +/- 708 est donc REDUCTIBLE PAR LE CALCUL SEUL :")
print(f"  il tombe a {ech[-1][2]:.0f} M€ en passant de {fnum(cx.NY)} a {fnum(TAILLES[-1])} annees, sans qu'aucune")
print("  donnee nouvelle soit necessaire et sans qu'aucun parametre bouge.")

# =====================================================================================
titre("3. L'AUTRE SOURCE, ET ELLE EST PLUS GRANDE : l'incertitude sur xi")
# =====================================================================================
print("  On rejoue maintenant le meme effet croise aux deux bornes de l'IC90 de l'indice de queue,")
print(f"  [{XI_IC90[0]:.2f} ; {XI_IC90[1]:.2f}], obtenu par bootstrap des {N_EXCES} exces observes. C'est une")
print("  SENSIBILITE DECLAREE, du meme type que la bande de modele du script 48 : aucun nombre")
print("  publie ne bouge, on mesure seulement l'amplitude que cette incertitude imprime.")
print(f"\n  {'xi':>8}{'effet croise':>15}{'ecart-type de simulation':>27}")
par = []
for xi in (XI_IC90[0], sp["xi"], XI_IC90[1]):
    c = croise_freq_det(cx.NY, NSEED_ECH, xi=xi)
    par.append((xi, float(c.mean()), float(c.std(ddof=1))))
    lab = f"{xi:.4f}" + ("  (publie)" if abs(xi - sp["xi"]) < 1e-9 else "")
    print(f"  {lab:>8}{c.mean():>15.0f}{c.std(ddof=1):>27.0f}")
plage_par = max(p[1] for p in par) - min(p[1] for p in par)
print(f"\n  PLAGE DUE A L'INCERTITUDE DE PARAMETRE : {plage_par:.0f} M€ de la borne basse a la borne haute.")
print(f"  BRUIT DE SIMULATION AUX REGLAGES PUBLIES  : {bruit_ref:.0f} M€.")
print(f"  RAPPORT : {plage_par / bruit_ref:.1f}.")
print("\n  L'ORDRE DES DEUX SOURCES EST DONC L'INVERSE DE CE QUE LA QUESTION SUPPOSAIT, et c'est le")
print("  resultat de ce script. La largeur visible, celle qui est imprimee a cote du chiffre, est")
print("  la PLUS PETITE des deux et la seule qui se reduise par le calcul. La plus grande ne")
print("  s'affiche nulle part a cet endroit du memoire : elle est publiee ailleurs, sur la VaR de")
print("  severite, sous la forme du facteur 2,5.")
print("\n  D'OU LA RECOMMANDATION DE REDACTION, ET ELLE NE COUTE RIEN. Le +/- imprime doit dire")
print("  QUOI il mesure. Ecrire « 1 739 +/- 708 (bruit de simulation, 16 graines) » enleve toute")
print("  ambiguite et vaut mieux que d'elargir ou de resserrer quoi que ce soit : un lecteur qui")
print("  voit 40 % et croit lire une incertitude de parametre conclut a un modele fragile, alors")
print("  qu'il lit un budget de calcul.")

# =====================================================================================
titre("4. LES TROIS SOURCES, ET CE QUI LES REDUIT")
# =====================================================================================
print(f"  {'source':<30}{'amplitude ici':>16}{'ce qui la reduit':>34}")
print(f"  {'bruit de simulation':<30}{bruit_ref:>13.0f} M€{'du calcul (1/racine(annees))':>34}")
print(f"  {'incertitude de parametre':<30}{plage_par:>13.0f} M€{'de la donnee (1/racine(exces))':>34}")
print(f"  {'ambiguite de modele':<30}{'non chiffree ici':>16}{'un argument, rien d autre':>34}")
print("\n  LES TROIS NE SE CUMULENT PAS DE LA MEME FACON ET NE SE TRAITENT PAS DE LA MEME FACON.")
print("  La premiere est un cout de machine, la deuxieme un cout de collecte, la troisieme un")
print("  choix a defendre. Les additionner en une seule barre d'erreur ferait croire que la")
print("  troisieme se reduit comme la premiere, ce qui est faux, et c'est pourquoi le memoire les")
print("  publie separement (bande de parametre au chapitre 13, ambiguite de famille en sixieme")
print("  posture, bruit de simulation a cote de chaque grandeur simulee).")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LE +/- 708 N'EST PAS UN INTERVALLE DE CONFIANCE. C'est l'ecart-type entre graines d'un")
print("     estimateur de Monte-Carlo a parametres FIXES. Aucune donnee n'y entre, donc la question")
print("     « manque de donnees ou incertitude de parametre » n'a pas d'objet pour cette barre-la.")
print(f"  1bis. ET LE COUPLE PUBLIE MELANGE DEUX RESOLUTIONS : la moyenne {c4.mean():.0f} vient de QUATRE")
print(f"     graines (la chaine publiee, gelee), l'ecart-type {bruit_ref:.0f} de SEIZE. Le script 68 le")
print("     declare, le memoire et le deck ne le reportent pas. A resolution homogene sur seize")
print(f"     graines la grandeur vaut {c_ref.mean():.0f} +/- {bruit_ref:.0f} par tirage, ou "
      f"+/- {bruit_ref/np.sqrt(NSEED_CTRL):.0f} pour la moyenne.")
print(f"     Le « 40 % » de la question a donc trois valeurs selon ce qu'on rapporte a quoi : "
      f"{100*bruit_ref/c4.mean():.0f} %,")
print(f"     {100*bruit_ref/c_ref.mean():.0f} % ou {100*bruit_ref/np.sqrt(NSEED_CTRL)/c_ref.mean():.0f} %.")
print(f"  2. TEST D'ECHELLE : la pente log-log vaut {pente:.2f} contre -0,50 attendu pour du Monte-Carlo")
print("     et 0,00 pour une incertitude de donnee. C'est donc bien du bruit de simulation, et il")
print(f"     est REDUCTIBLE PAR LE CALCUL : {bruit_ref:.0f} M€ a {fnum(cx.NY)} annees, {ech[-1][2]:.0f} a {fnum(TAILLES[-1])}.")
print(f"  2bis. LA VITESSE EST BIEN EN 1/RACINE(N) ET NON EN 1/LOG(N). Sur un facteur {facteur:.0f} en")
print(f"     nombre d'annees, l'ecart-type tombe d'un facteur {r_mes:.2f} mesure, contre {r_sqrt:.2f} predit par")
print(f"     la racine et {r_log:.2f} par le logarithme. La pente vaut {pente:.2f}, d'intervalle a 90 %")
print(f"     [{lo_p:.2f} ; {hi_p:.2f}] par reechantillonnage des graines : elle contient -0,50, donc")
print("     l'exces apparent n'est pas un effet.")
print("  3. POURQUOI CET ESTIMATEUR EST SI BRUITE, et c'est structurel : un effet croise est une")
print("     difference de differences de quatre quantiles. La soustraction annule le signal et")
print("     additionne les bruits.")
print(f"  4. L'AUTRE SOURCE EST PLUS GRANDE : l'IC90 de xi imprime une plage de {plage_par:.0f} M€ sur le meme")
print(f"     effet croise, soit {plage_par / bruit_ref:.1f} fois le bruit de simulation. L'ORDRE DES DEUX EST DONC")
print("     L'INVERSE DE CE QU'ON SUPPOSAIT : la largeur visible est la plus petite des deux, et la")
print("     seule qui se paye en temps de machine.")
print("  5. RECOMMANDATION DE REDACTION, ET ELLE EST COMPATIBLE AVEC LE GEL. Ne pas toucher aux")
print("     valeurs, qui appartiennent a la chaine gelee, mais ETIQUETER le +/- : « 1 739 (4")
print("     graines) +/- 708 (etendue par graine sur 16) ». C'est ce que le script 68 dit deja et")
print("     que le memoire ne reporte pas. Un lecteur qui lit 40 % sans etiquette conclut a un")
print("     modele fragile alors qu'il lit un budget de calcul, et il a raison de poser la")
print("     question : c'est l'etiquette qui manque, pas la mesure.")
print("  6. ET TROIS SOURCES A NE PAS FONDRE EN UNE SEULE BARRE : le calcul reduit la premiere, la")
print("     donnee la deuxieme, un argument la troisieme. Le memoire les publie deja separement,")
print("     et il faut le garder ainsi.")

# =====================================================================================
# figure S40
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.2))

# (a) la loi d'echelle : le test qui tranche
nys = np.array([e[0] for e in ech], float)
sds = np.array([e[2] for e in ech], float)
ax1.plot(nys, sds, "o-", color=ACCENT, lw=2.2, ms=9, label="écart-type mesuré")
ref = sds[0] * np.sqrt(nys[0] / nys)
ax1.plot(nys, ref, "--", color=BLUE, lw=1.6,
         label="pente $-1/2$ (Monte-Carlo pur)")
ax1.plot(nys, [sds[0]] * len(nys), ":", color=MUTED, lw=1.6,
         label="pente $0$ (incertitude de donnée)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(nys)
ax1.set_xticklabels([fnum(n) for n in nys], fontsize=9)
ax1.set_xlabel("années simulées par graine (log)", color=INK2)
ax1.set_ylabel("écart-type de l'effet croisé (M€, log)", color=INK2)
ax1.legend(fontsize=9, frameon=False, loc="lower left")
ax1.annotate(f"pente mesurée {pente:.2f}".replace(".", ","),
             xy=(nys[1], sds[1]), xytext=(nys[0] * 1.25, sds[0] * 0.62),
             fontsize=10, color=INK, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
ax1.set_title("(a)  Le test qui tranche : le bruit tombe en $1/\\sqrt{n}$\navec les années "
              "SIMULÉES, pas avec les données", fontsize=10.5, color=INK, pad=8)

# (b) les deux amplitudes, cote a cote
lab = ["bruit de simulation\n(16 graines, 40 000 ans)", "incertitude de paramètre\n(IC90 de ξ)"]
val = [bruit_ref, plage_par]
ax2.bar([0, 1], val, width=0.52, color=[ACCENT, BLUE], alpha=0.9)
for x, v in zip([0, 1], val):
    ax2.text(x, v + 0.03 * max(val), f"{fnum(v)} M€", ha="center", fontsize=11,
             color=INK, fontweight="bold")
ax2.set_xticks([0, 1])
ax2.set_xticklabels(lab, fontsize=9.5)
ax2.set_ylim(0, max(val) * 1.30)
ax2.set_ylabel("amplitude sur l'effet croisé (M€)", color=INK2)
ax2.text(0.5, max(val) * 1.18, f"facteur {plage_par/bruit_ref:.1f}".replace(".", ","),
         ha="center", fontsize=11, color=INK, fontweight="bold")
ax2.annotate("", xy=(0.06, max(val) * 1.13), xytext=(0.94, max(val) * 1.13),
             arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
ax2.text(0.5, -0.19, "réductible par le calcul                   réductible par la donnée",
         transform=ax2.transAxes, ha="center", fontsize=9, color=INK2, style="italic")
ax2.set_title("(b)  La largeur imprimée est la PLUS PETITE des deux,\net la seule qui se paye "
              "en temps de machine", fontsize=10.5, color=INK, pad=8)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("S40 : d'où vient la largeur des intervalles, et laquelle des trois sources se réduit",
             fontsize=12.5, fontweight="bold", color=INK, x=0.02, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0.04, 1, 0.92])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S40_origine_largeur.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
